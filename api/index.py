def app(environ, start_response):
    status = "200 OK"
    headers = [("Content-Type", "text/plain; charset=utf-8")]
    body = "Micro-Milling App is running"
    start_response(status, headers)
    return [body.encode("utf-8")]


application = app
