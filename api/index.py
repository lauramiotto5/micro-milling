def app(environ, start_response):
    status = "200 OK"
    headers = [("Content-Type", "text/plain; charset=utf-8")]
    body = b"Micro-Milling App is running"
    start_response(status, headers)
    return [body]


application = app
