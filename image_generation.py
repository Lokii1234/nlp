from VortexGPT import Client
from PIL import Image
from io import BytesIO
def generate_and_show_image(prompt):
    try:
        resp = Client.create_generation("prodia", prompt)
        Image.open(BytesIO(resp)).show()
        print(f"🤖: Image shown.")
    except Exception as e:
        print(f"🤖: {e}")
