import numpy as np
import cv2
import mahotas
from scipy.ndimage import label

def compute_zernike_moments(image):
    """Compute Zernike moments for a binary image."""
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        (x, y), radius = cv2.minEnclosingCircle(contours[0])
        radius = int(radius)
        zm = mahotas.features.zernike_moments(binary_image, radius, degree=8)
        return zm
    else:
        return None

def separate_objects(mask):
    """Separate objects in a binary mask."""
    labeled_mask, num_features = label(mask)
    separated_images = []

    for i in range(1, num_features + 1):
        object_image = np.zeros_like(mask, dtype=np.uint8)
        object_image[labeled_mask == i] = 255
        separated_images.append(object_image)

    return separated_images

def compute_distance(zm1, zm2):
    """Compute the Euclidean distance between two Zernike moments."""
    return np.linalg.norm(np.array(zm1) - np.array(zm2))

def annotate_object(image, text="Object Found!"):
    """Annotate the image with the given text."""
    annotated_image = image.copy()
    cv2.putText(annotated_image, text, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (170, 255, 0), 2)
    return annotated_image

def find_true_object(sample_zernike, image_path):
    """Find the true object in an image based on the given sample Zernike moment."""
    # Read the binary mask
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Binarize the mask
    _, binary_mask = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Separate objects
    separated_images = separate_objects(binary_mask)

    annotated_images = []

    for idx, img in enumerate(separated_images):
        zm = compute_zernike_moments(img)
        if zm is not None:
            distance = compute_distance(sample_zernike, zm)
            if distance < 0.15:
                annotated_img = annotate_object(img, "focused RBC")
            elif distance <0.35:
                annotated_img = annotate_object(img, "unfocused RBC")
            elif distance <0.7:
                annotated_img = annotate_object(img, "WBC")
            else:
                annotated_img = annotate_object(img, "something else")
            annotated_images.append(annotated_img)
    return annotated_images

# Example usage
mask = cv2.imread('separated_object_example.png', cv2.IMREAD_GRAYSCALE)
sample_zernike = compute_zernike_moments(mask) # Replace with the actual Zernike moments
image_path = 'masks/ERY_030_t00_selected_frames_01355191.png'

annotated_images = find_true_object(sample_zernike, image_path)

# Save or display the annotated images
for idx, img in enumerate(annotated_images):
    cv2.imwrite(f'annotated_separated_object_{idx+1}.png', img)
    cv2.imshow(f'Annotated Separated Object {idx+1}', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
