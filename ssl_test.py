import ssl
import socket
import certifi

hostname = 'smtp.gmail.com'
context = ssl.create_default_context()
context.load_verify_locations(cafile=certifi.where())

with socket.create_connection((hostname, 465)) as sock:
    with context.wrap_socket(sock, server_hostname=hostname) as ssock:
        print("SSL handshake successful:", ssock.version())
