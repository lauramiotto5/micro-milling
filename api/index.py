def handler(request):
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "text/plain"},
        "body": "Micro-Milling App is running"
    }


app = handler
application = handler
