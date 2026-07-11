"""
Vercel serverless handler for the Micro-Milling application
"""

def handler(request):
    """
    Handler for Vercel serverless functions.
    Required for Vercel to recognize this as a Python function.
    """
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'text/plain'},
        'body': 'Micro-Milling App is running'
    }

# Export for Vercel
app = handler
application = handler
