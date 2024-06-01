import numpy as np
import base64
import cv2
import torchvision.transforms as transforms


def extract_feature(img_path, transform, model):
    img = cv2.imread(img_path)
    retval, buffer = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    image_base64 = base64.b64encode(buffer)

    img = transform(img)
    img = img.view(1, img.shape[0], img.shape[1], img.shape[2])
    embedded_vec = model(img)['embedding'] .view(-1).detach().numpy().tolist()
    return embedded_vec, image_base64


def transform(image_height, image_width):
    def any_to_rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transformation = transforms.Compose([
                     transforms.Lambda(any_to_rgb),
                     transforms.ToTensor(),

                     transforms.Resize((image_height, image_width)),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                        ])
    return transformation


def get_image(img_base64, transformations, client, model):
    decoded_image = base64.b64decode(img_base64)
    np_arr = np.frombuffer(decoded_image, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img = transformations(img)
    img = img.view(1, img.shape[0], img.shape[1], img.shape[2])
    embedded_vec = model(img)['embedding'] .view(-1).detach().numpy().tolist()
    hits = client.search(
        collection_name="Image-Search",
        query_vector=embedded_vec,
        limit=10,
    )
    results = [{"image": hits[i].payload["base64"], "label": hits[i].payload["label"],
                "score": hits[i].score} for i, hit in enumerate(hits)]

    return results
