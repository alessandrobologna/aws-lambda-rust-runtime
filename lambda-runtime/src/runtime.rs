use crate::{
    layers::{CatchPanicService, RuntimeApiClientService, RuntimeApiResponseService},
    requests::{IntoRequest, NextEventRequest},
    types::{invoke_request_id, IntoFunctionResponse, LambdaEvent},
    Config, Context, Diagnostic,
};
use futures::stream::FuturesUnordered;
use http_body_util::BodyExt;
use lambda_runtime_api_client::{BoxError, Client as ApiClient};
use serde::{Deserialize, Serialize};
use std::{
    env,
    fmt::Debug,
    future::Future,
    sync::{Arc, OnceLock},
};
use tokio::sync::watch;
use tokio_stream::{Stream, StreamExt};
use tower::{Layer, Service, ServiceExt};
use tracing::{error, trace, warn};

/* ----------------------------------------- INVOCATION ---------------------------------------- */

/// A simple container that provides information about a single invocation of a Lambda function.
pub struct LambdaInvocation {
    /// The header of the request sent to invoke the Lambda function.
    pub parts: http::response::Parts,
    /// The body of the request sent to invoke the Lambda function.
    pub body: bytes::Bytes,
    /// The context of the Lambda invocation.
    pub context: Context,
}

/* ------------------------------------------ RUNTIME ------------------------------------------ */

/// Lambda runtime executing a handler function on incoming requests.
///
/// Middleware can be added to a runtime using the [Runtime::layer] method in order to execute
/// logic prior to processing the incoming request and/or after the response has been sent back
/// to the Lambda Runtime API.
///
/// # Example
/// ```no_run
/// use lambda_runtime::{Error, LambdaEvent, Runtime};
/// use serde_json::Value;
/// use tower::service_fn;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Error> {
///     let func = service_fn(func);
///     Runtime::new(func).run().await?;
///     Ok(())
/// }
///
/// async fn func(event: LambdaEvent<Value>) -> Result<Value, Error> {
///     Ok(event.payload)
/// }
/// ````
pub struct Runtime<S> {
    service: S,
    config: Arc<Config>,
    client: Arc<ApiClient>,
}

/// One-time marker to log X-Ray behavior in concurrent mode.
static XRAY_LOGGED: OnceLock<()> = OnceLock::new();

impl<F, EventPayload, Response, BufferedResponse, StreamingResponse, StreamItem, StreamError>
    Runtime<
        RuntimeApiClientService<
            RuntimeApiResponseService<
                CatchPanicService<'_, F>,
                EventPayload,
                Response,
                BufferedResponse,
                StreamingResponse,
                StreamItem,
                StreamError,
            >,
        >,
    >
where
    F: Service<LambdaEvent<EventPayload>, Response = Response>,
    F::Future: Future<Output = Result<Response, F::Error>>,
    F::Error: Into<Diagnostic> + Debug,
    EventPayload: for<'de> Deserialize<'de>,
    Response: IntoFunctionResponse<BufferedResponse, StreamingResponse>,
    BufferedResponse: Serialize,
    StreamingResponse: Stream<Item = Result<StreamItem, StreamError>> + Unpin + Send + 'static,
    StreamItem: Into<bytes::Bytes> + Send,
    StreamError: Into<BoxError> + Send + Debug,
{
    /// Create a new runtime that executes the provided handler for incoming requests.
    ///
    /// In order to start the runtime and poll for events on the [Lambda Runtime
    /// APIs](https://docs.aws.amazon.com/lambda/latest/dg/runtimes-api.html), you must call
    /// [Runtime::run].
    ///
    /// Note that manually creating a [Runtime] does not add tracing to the executed handler
    /// as is done by [super::run]. If you want to add the default tracing functionality, call
    /// [Runtime::layer] with a [super::layers::TracingLayer].
    pub fn new(handler: F) -> Self {
        trace!("Loading config from env");
        let config = Arc::new(Config::from_env());
        let pool_size = config.max_concurrency.unwrap_or(1).max(1) as usize;
        let client = Arc::new(
            ApiClient::builder()
                .with_pool_size(pool_size)
                .build()
                .expect("Unable to create a runtime client"),
        );
        Self {
            service: wrap_handler(handler, client.clone()),
            config,
            client,
        }
    }
}

impl<S> Runtime<S> {
    /// Add a new layer to this runtime. For an incoming request, this layer will be executed
    /// before any layer that has been added prior.
    ///
    /// # Example
    /// ```no_run
    /// use lambda_runtime::{layers, Error, LambdaEvent, Runtime};
    /// use serde_json::Value;
    /// use tower::service_fn;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Error> {
    ///     let runtime = Runtime::new(service_fn(echo)).layer(
    ///         layers::TracingLayer::new()
    ///     );
    ///     runtime.run().await?;
    ///     Ok(())
    /// }
    ///
    /// async fn echo(event: LambdaEvent<Value>) -> Result<Value, Error> {
    ///     Ok(event.payload)
    /// }
    /// ```
    pub fn layer<L>(self, layer: L) -> Runtime<L::Service>
    where
        L: Layer<S>,
        L::Service: Service<LambdaInvocation, Response = (), Error = BoxError>,
    {
        Runtime {
            client: self.client,
            config: self.config,
            service: layer.layer(self.service),
        }
    }
}

impl<S> Runtime<S>
where
    S: Service<LambdaInvocation, Response = (), Error = BoxError> + Clone + Send + 'static,
    S::Future: Send,
{
    /// Start the runtime in concurrent mode when configured for Lambda managed-concurrency.
    ///
    /// If `AWS_LAMBDA_MAX_CONCURRENCY` is not set or is `<= 1`, this falls back to the
    /// sequential `run_with_incoming` loop so that the same handler can run on both
    /// classic Lambda and Lambda Managed Instances.
    pub async fn run_concurrent(self) -> Result<(), BoxError> {
        if self.config.is_concurrent() {
            let max_concurrency = self.config.max_concurrency.unwrap_or(1);
            Self::run_concurrent_inner(self.service, self.config, self.client, max_concurrency).await
        } else {
            let incoming = incoming(&self.client);
            Self::run_with_incoming(self.service, self.config, incoming).await
        }
    }

    /// Concurrent processing using N independent long-poll loops (for Lambda managed-concurrency).
    async fn run_concurrent_inner(
        service: S,
        config: Arc<Config>,
        client: Arc<ApiClient>,
        max_concurrency: u32,
    ) -> Result<(), BoxError> {
        let limit = max_concurrency as usize;
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        let mut workers = FuturesUnordered::new();
        for _ in 1..limit {
            workers.push(tokio::spawn(concurrent_worker_loop(
                service.clone(),
                config.clone(),
                client.clone(),
                shutdown_rx.clone(),
            )));
        }
        workers.push(tokio::spawn(concurrent_worker_loop(
            service,
            config.clone(),
            client.clone(),
            shutdown_rx.clone(),
        )));

        let mut first_error: Option<BoxError> = None;
        while let Some(result) = futures::StreamExt::next(&mut workers).await {
            match result {
                Ok(Ok(())) => {}
                Ok(Err(err)) => {
                    if first_error.is_none() {
                        error!(error = %err, "Concurrent worker exited with error; shutting down");
                        let _ = shutdown_tx.send(true);
                        first_error = Some(err);
                    } else {
                        error!(error = %err, "Concurrent worker exited with error");
                    }
                }
                Err(join_err) => {
                    let err: BoxError = Box::new(join_err);
                    if first_error.is_none() {
                        error!(error = %err, "Concurrent worker panicked; shutting down");
                        let _ = shutdown_tx.send(true);
                        first_error = Some(err);
                    } else {
                        error!(error = %err, "Concurrent worker panicked");
                    }
                }
            }
        }

        match first_error {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

impl<S> Runtime<S>
where
    S: Service<LambdaInvocation, Response = (), Error = BoxError>,
{
    /// Start the runtime and begin polling for events on the Lambda Runtime API.
    pub async fn run(self) -> Result<(), BoxError> {
        let incoming = incoming(&self.client);
        Self::run_with_incoming(self.service, self.config, incoming).await
    }

    /// Internal utility function to start the runtime with a customized incoming stream.
    /// This implements the core of the [Runtime::run] method.
    pub(crate) async fn run_with_incoming(
        mut service: S,
        config: Arc<Config>,
        incoming: impl Stream<Item = Result<http::Response<hyper::body::Incoming>, BoxError>> + Send,
    ) -> Result<(), BoxError> {
        tokio::pin!(incoming);
        while let Some(next_event_response) = incoming.next().await {
            trace!("New event arrived (run loop)");
            let event = next_event_response?;
            drive_event_loop_iteration(&mut service, &config, event, true).await?;
        }
        Ok(())
    }
}

/* ------------------------------------------- UTILS ------------------------------------------- */

#[allow(clippy::type_complexity)]
fn wrap_handler<'a, F, EventPayload, Response, BufferedResponse, StreamingResponse, StreamItem, StreamError>(
    handler: F,
    client: Arc<ApiClient>,
) -> RuntimeApiClientService<
    RuntimeApiResponseService<
        CatchPanicService<'a, F>,
        EventPayload,
        Response,
        BufferedResponse,
        StreamingResponse,
        StreamItem,
        StreamError,
    >,
>
where
    F: Service<LambdaEvent<EventPayload>, Response = Response>,
    F::Future: Future<Output = Result<Response, F::Error>>,
    F::Error: Into<Diagnostic> + Debug,
    EventPayload: for<'de> Deserialize<'de>,
    Response: IntoFunctionResponse<BufferedResponse, StreamingResponse>,
    BufferedResponse: Serialize,
    StreamingResponse: Stream<Item = Result<StreamItem, StreamError>> + Unpin + Send + 'static,
    StreamItem: Into<bytes::Bytes> + Send,
    StreamError: Into<BoxError> + Send + Debug,
{
    let safe_service = CatchPanicService::new(handler);
    let response_service = RuntimeApiResponseService::new(safe_service);
    RuntimeApiClientService::new(response_service, client)
}

fn incoming(
    client: &ApiClient,
) -> impl Stream<Item = Result<http::Response<hyper::body::Incoming>, BoxError>> + Send + '_ {
    async_stream::stream! {
        loop {
            trace!("Waiting for next event (incoming loop)");
            let req = NextEventRequest.into_req().expect("Unable to construct request");
            let res = client.call(req).await;
            yield res;
        }
    }
}

/// Creates a future that polls the `/next` endpoint.
async fn next_event_future(client: Arc<ApiClient>) -> Result<http::Response<hyper::body::Incoming>, BoxError> {
    let req = NextEventRequest.into_req()?;
    client.call(req).await
}

async fn concurrent_worker_loop<S>(
    mut service: S,
    config: Arc<Config>,
    client: Arc<ApiClient>,
    mut shutdown_rx: watch::Receiver<bool>,
) -> Result<(), BoxError>
where
    S: Service<LambdaInvocation, Response = (), Error = BoxError>,
    S::Future: Send,
{
    loop {
        if *shutdown_rx.borrow() {
            break;
        }

        tokio::select! {
            changed = shutdown_rx.changed() => {
                match changed {
                    Ok(()) => {
                        if *shutdown_rx.borrow() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
            result = next_event_future(client.clone()) => {
                let event = match result {
                    Ok(event) => event,
                    Err(e) => {
                        warn!(error = %e, "Error polling /next, retrying");
                        continue;
                    }
                };

                drive_event_loop_iteration(&mut service, &config, event, false).await?;
            }
        }
    }

    Ok(())
}

async fn drive_event_loop_iteration<S>(
    service: &mut S,
    config: &Arc<Config>,
    event: http::Response<hyper::body::Incoming>,
    set_amzn_trace_env: bool,
) -> Result<(), BoxError>
where
    S: Service<LambdaInvocation, Response = (), Error = BoxError>,
{
    let (parts, incoming) = event.into_parts();

    #[cfg(debug_assertions)]
    if parts.status == http::StatusCode::NO_CONTENT {
        // Ignore the event if the status code is 204.
        // This is a way to keep the runtime alive when
        // there are no events pending to be processed.
        return Ok(());
    }

    // Build the invocation such that it can be sent to the service right away
    // when it is ready
    let body = incoming.collect().await?.to_bytes();
    let context = Context::new(invoke_request_id(&parts.headers)?, config.clone(), &parts.headers)?;
    let invocation = LambdaInvocation { parts, body, context };

    if set_amzn_trace_env {
        // Setup Amazon's default tracing data
        amzn_trace_env(&invocation.context);
    } else {
        // Inform users that X-Ray is available via context, not env var, in concurrent mode.
        XRAY_LOGGED.get_or_init(|| {
            trace!("Concurrent mode: _X_AMZN_TRACE_ID is not set; use context.xray_trace_id");
        });
    }

    // Wait for service to be ready
    let ready = service.ready().await?;

    // Once ready, call the service which will respond to the Lambda runtime API
    ready.call(invocation).await?;
    Ok(())
}

fn amzn_trace_env(ctx: &Context) {
    match &ctx.xray_trace_id {
        Some(trace_id) => env::set_var("_X_AMZN_TRACE_ID", trace_id),
        None => env::remove_var("_X_AMZN_TRACE_ID"),
    }
}

/* --------------------------------------------------------------------------------------------- */
/*                                             TESTS                                             */
/* --------------------------------------------------------------------------------------------- */

#[cfg(test)]
mod endpoint_tests {
    use super::{incoming, wrap_handler};
    use crate::{
        requests::{EventCompletionRequest, EventErrorRequest, IntoRequest, NextEventRequest},
        Config, Diagnostic, Error, Runtime,
    };
    use futures::future::BoxFuture;
    use http::{HeaderValue, StatusCode};
    use http_body_util::BodyExt;
    use httpmock::prelude::*;

    use lambda_runtime_api_client::Client;
    use std::{env, sync::Arc};
    use tokio_stream::StreamExt;

    #[tokio::test]
    async fn test_next_event() -> Result<(), Error> {
        let server = MockServer::start();
        let request_id = "156cb537-e2d4-11e8-9b34-d36013741fb9";
        let deadline = "1542409706888";

        let mock = server.mock(|when, then| {
            when.method(GET).path("/2018-06-01/runtime/invocation/next");
            then.status(200)
                .header("content-type", "application/json")
                .header("lambda-runtime-aws-request-id", request_id)
                .header("lambda-runtime-deadline-ms", deadline)
                .body("{}");
        });

        let base = server.base_url().parse().expect("Invalid mock server Uri");
        let client = Client::builder().with_endpoint(base).build()?;

        let req = NextEventRequest.into_req()?;
        let rsp = client.call(req).await.expect("Unable to send request");

        mock.assert_async().await;
        assert_eq!(rsp.status(), StatusCode::OK);
        assert_eq!(
            rsp.headers()["lambda-runtime-aws-request-id"],
            &HeaderValue::from_static(request_id)
        );
        assert_eq!(
            rsp.headers()["lambda-runtime-deadline-ms"],
            &HeaderValue::from_static(deadline)
        );

        let body = rsp.into_body().collect().await?.to_bytes();
        assert_eq!("{}", std::str::from_utf8(&body)?);
        Ok(())
    }

    #[tokio::test]
    async fn test_ok_response() -> Result<(), Error> {
        let server = MockServer::start();

        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/2018-06-01/runtime/invocation/156cb537-e2d4-11e8-9b34-d36013741fb9/response")
                .body("\"{}\"");
            then.status(200).body("");
        });

        let base = server.base_url().parse().expect("Invalid mock server Uri");
        let client = Client::builder().with_endpoint(base).build()?;

        let req = EventCompletionRequest::new("156cb537-e2d4-11e8-9b34-d36013741fb9", "{}");
        let req = req.into_req()?;

        let rsp = client.call(req).await?;

        mock.assert_async().await;
        assert_eq!(rsp.status(), StatusCode::OK);
        Ok(())
    }

    #[tokio::test]
    async fn test_error_response() -> Result<(), Error> {
        let diagnostic = Diagnostic {
            error_type: "InvalidEventDataError".into(),
            error_message: "Error parsing event data".into(),
        };
        let body = serde_json::to_string(&diagnostic)?;

        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/2018-06-01/runtime/invocation/156cb537-e2d4-11e8-9b34-d36013741fb9/error")
                .header("lambda-runtime-function-error-type", "unhandled")
                .body(body);
            then.status(200).body("");
        });

        let base = server.base_url().parse().expect("Invalid mock server Uri");
        let client = Client::builder().with_endpoint(base).build()?;

        let req = EventErrorRequest {
            request_id: "156cb537-e2d4-11e8-9b34-d36013741fb9",
            diagnostic,
        };
        let req = req.into_req()?;
        let rsp = client.call(req).await?;

        mock.assert_async().await;
        assert_eq!(rsp.status(), StatusCode::OK);
        Ok(())
    }

    #[tokio::test]
    async fn successful_end_to_end_run() -> Result<(), Error> {
        let server = MockServer::start();
        let request_id = "156cb537-e2d4-11e8-9b34-d36013741fb9";
        let deadline = "1542409706888";

        let next_request = server.mock(|when, then| {
            when.method(GET).path("/2018-06-01/runtime/invocation/next");
            then.status(200)
                .header("content-type", "application/json")
                .header("lambda-runtime-aws-request-id", request_id)
                .header("lambda-runtime-deadline-ms", deadline)
                .body("{}");
        });
        let next_response = server.mock(|when, then| {
            when.method(POST)
                .path(format!("/2018-06-01/runtime/invocation/{request_id}/response"))
                .body("{}");
            then.status(200).body("");
        });

        let base = server.base_url().parse().expect("Invalid mock server Uri");
        let client = Client::builder().with_endpoint(base).build()?;

        async fn func(event: crate::LambdaEvent<serde_json::Value>) -> Result<serde_json::Value, Error> {
            let (event, _) = event.into_parts();
            Ok(event)
        }
        let f = crate::service_fn(func);

        // set env vars needed to init Config if they are not already set in the environment
        if env::var("AWS_LAMBDA_RUNTIME_API").is_err() {
            env::set_var("AWS_LAMBDA_RUNTIME_API", server.base_url());
        }
        if env::var("AWS_LAMBDA_FUNCTION_NAME").is_err() {
            env::set_var("AWS_LAMBDA_FUNCTION_NAME", "test_fn");
        }
        if env::var("AWS_LAMBDA_FUNCTION_MEMORY_SIZE").is_err() {
            env::set_var("AWS_LAMBDA_FUNCTION_MEMORY_SIZE", "128");
        }
        if env::var("AWS_LAMBDA_FUNCTION_VERSION").is_err() {
            env::set_var("AWS_LAMBDA_FUNCTION_VERSION", "1");
        }
        if env::var("AWS_LAMBDA_LOG_STREAM_NAME").is_err() {
            env::set_var("AWS_LAMBDA_LOG_STREAM_NAME", "test_stream");
        }
        if env::var("AWS_LAMBDA_LOG_GROUP_NAME").is_err() {
            env::set_var("AWS_LAMBDA_LOG_GROUP_NAME", "test_log");
        }
        let config = Config::from_env();

        let client = Arc::new(client);
        let runtime = Runtime {
            client: client.clone(),
            config: Arc::new(config),
            service: wrap_handler(f, client),
        };
        let client = &runtime.client;
        let incoming = incoming(client).take(1);
        Runtime::run_with_incoming(runtime.service, runtime.config, incoming).await?;

        next_request.assert_async().await;
        next_response.assert_async().await;
        Ok(())
    }

    async fn run_panicking_handler<F>(func: F) -> Result<(), Error>
    where
        F: FnMut(crate::LambdaEvent<serde_json::Value>) -> BoxFuture<'static, Result<serde_json::Value, Error>>
            + Send
            + 'static,
    {
        let server = MockServer::start();
        let request_id = "156cb537-e2d4-11e8-9b34-d36013741fb9";
        let deadline = "1542409706888";

        let next_request = server.mock(|when, then| {
            when.method(GET).path("/2018-06-01/runtime/invocation/next");
            then.status(200)
                .header("content-type", "application/json")
                .header("lambda-runtime-aws-request-id", request_id)
                .header("lambda-runtime-deadline-ms", deadline)
                .body("{}");
        });

        let next_response = server.mock(|when, then| {
            when.method(POST)
                .path(format!("/2018-06-01/runtime/invocation/{request_id}/error"))
                .header("lambda-runtime-function-error-type", "unhandled");
            then.status(200).body("");
        });

        let base = server.base_url().parse().expect("Invalid mock server Uri");
        let client = Client::builder().with_endpoint(base).build()?;

        let f = crate::service_fn(func);

        let config = Arc::new(Config {
            function_name: "test_fn".to_string(),
            memory: 128,
            version: "1".to_string(),
            log_stream: "test_stream".to_string(),
            log_group: "test_log".to_string(),
            max_concurrency: None,
        });

        let client = Arc::new(client);
        let runtime = Runtime {
            client: client.clone(),
            config,
            service: wrap_handler(f, client),
        };
        let client = &runtime.client;
        let incoming = incoming(client).take(1);
        Runtime::run_with_incoming(runtime.service, runtime.config, incoming).await?;

        next_request.assert_async().await;
        next_response.assert_async().await;
        Ok(())
    }

    #[tokio::test]
    async fn panic_in_async_run() -> Result<(), Error> {
        run_panicking_handler(|_| Box::pin(async { panic!("This is intentionally here") })).await
    }

    #[tokio::test]
    async fn panic_outside_async_run() -> Result<(), Error> {
        run_panicking_handler(|_| {
            panic!("This is intentionally here");
        })
        .await
    }

    #[test]
    fn config_parses_max_concurrency() {
        // Preserve existing env values
        let prev_fn = env::var("AWS_LAMBDA_FUNCTION_NAME").ok();
        let prev_mem = env::var("AWS_LAMBDA_FUNCTION_MEMORY_SIZE").ok();
        let prev_ver = env::var("AWS_LAMBDA_FUNCTION_VERSION").ok();
        let prev_log_stream = env::var("AWS_LAMBDA_LOG_STREAM_NAME").ok();
        let prev_log_group = env::var("AWS_LAMBDA_LOG_GROUP_NAME").ok();
        let prev_max = env::var("AWS_LAMBDA_MAX_CONCURRENCY").ok();

        env::set_var("AWS_LAMBDA_FUNCTION_NAME", "test_fn");
        env::set_var("AWS_LAMBDA_FUNCTION_MEMORY_SIZE", "128");
        env::set_var("AWS_LAMBDA_FUNCTION_VERSION", "1");
        env::set_var("AWS_LAMBDA_LOG_STREAM_NAME", "test_stream");
        env::set_var("AWS_LAMBDA_LOG_GROUP_NAME", "test_log");
        env::set_var("AWS_LAMBDA_MAX_CONCURRENCY", "4");

        let cfg = Config::from_env();
        assert_eq!(cfg.max_concurrency, Some(4));
        assert!(cfg.is_concurrent());

        // Restore env
        if let Some(v) = prev_fn {
            env::set_var("AWS_LAMBDA_FUNCTION_NAME", v);
        } else {
            env::remove_var("AWS_LAMBDA_FUNCTION_NAME");
        }
        if let Some(v) = prev_mem {
            env::set_var("AWS_LAMBDA_FUNCTION_MEMORY_SIZE", v);
        } else {
            env::remove_var("AWS_LAMBDA_FUNCTION_MEMORY_SIZE");
        }
        if let Some(v) = prev_ver {
            env::set_var("AWS_LAMBDA_FUNCTION_VERSION", v);
        } else {
            env::remove_var("AWS_LAMBDA_FUNCTION_VERSION");
        }
        if let Some(v) = prev_log_stream {
            env::set_var("AWS_LAMBDA_LOG_STREAM_NAME", v);
        } else {
            env::remove_var("AWS_LAMBDA_LOG_STREAM_NAME");
        }
        if let Some(v) = prev_log_group {
            env::set_var("AWS_LAMBDA_LOG_GROUP_NAME", v);
        } else {
            env::remove_var("AWS_LAMBDA_LOG_GROUP_NAME");
        }
        if let Some(v) = prev_max {
            env::set_var("AWS_LAMBDA_MAX_CONCURRENCY", v);
        } else {
            env::remove_var("AWS_LAMBDA_MAX_CONCURRENCY");
        }
    }
}
