import numpy as np
import os
from PIL import Image

NUM_CHANNELS = 3


# --------------------------------------------------------------------------- #

def img_read_helper(path):
    """
    Creates an RGBImage object from the given image file
    """
    # Open the image in RGB
    img = Image.open(path).convert("RGB")
    # Convert to numpy array and then to a list
    matrix = np.array(img).tolist()
    # Use student's code to create an RGBImage object
    return RGBImage(matrix)


def img_save_helper(path, image):
    """
    Saves the given RGBImage instance to the given path
    """
    # Convert list to numpy array
    img_array = np.array(image.get_pixels())
    # Convert numpy array to PIL Image object
    img = Image.fromarray(img_array.astype(np.uint8))
    # Save the image object to path
    img.save(path)


# --------------------------------------------------------------------------- #

# Part 1: RGB Image #
class RGBImage:
    """
    Represents an image in RGB format
    """

    def __init__(self, pixels):
        """
        Initializes a new RGBImage object

        # Test with non-rectangular list
        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        # Test instance variables
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.pixels
        [[[255, 255, 255], [0, 0, 0]]]
        >>> img.num_rows
        1
        >>> img.num_cols
        2

        >>> pixels = {}
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        >>> pixels = []
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        >>> pixels = [[1, 2], 'a']
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        >>> pixels = [[1, 2], [1, 2, 3]]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        >>> pixels = [[[1, 2], 2]]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        >>> pixels = [[[1, 2, 3], [1, 2]]]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        >>> pixels = [[['a', 'b', 'c'], [1, 2, 3]]]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        ValueError

        >>> pixels = [[[-1, 2, 3], [1, 2, 3]]]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        ValueError
        """
        # YOUR CODE GOES HERE #
        # Raise exceptions here

        if not isinstance(pixels, list) or not pixels:
            raise TypeError()
        if not all([isinstance(row, list) for row in pixels]):
            raise TypeError()
        if not all([len(row) != 0 for row in pixels]):
            raise TypeError()
        if not all([len(row) == len(pixels[0]) for row in pixels]):
            raise TypeError()
        if not all([isinstance(pixel, list) for row in pixels for pixel in row]):
            raise TypeError()
        if not all([len(pixel) == 3 for row in pixels for pixel in row]):
            raise TypeError()
        if not all([isinstance(color, int) and 0 <= color <= 255 for row in pixels for pixel in row for color in pixel]):
            raise ValueError()


        self.pixels = pixels
        self.num_rows = len(pixels)
        self.num_cols = len(pixels[0])

    def size(self):
        """
        Returns the size of the image in (rows, cols) format

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 2)
        """
        # YOUR CODE GOES HERE #
        return (self.num_rows, self.num_cols)

    def get_pixels(self):
        """
        Returns a copy of the image pixel array

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_pixels = img.get_pixels()

        # Check if this is a deep copy
        >>> img_pixels                               # Check the values
        [[[255, 255, 255], [0, 0, 0]]]
        >>> id(pixels) != id(img_pixels)             # Check outer list
        True
        >>> id(pixels[0]) != id(img_pixels[0])       # Check row
        True
        >>> id(pixels[0][0]) != id(img_pixels[0][0]) # Check pixel
        True
        """
        # YOUR CODE GOES HERE #

        return [[[color for color in pixel] for pixel in row] for row in self.pixels]

    def copy(self):
        """
        Returns a copy of this RGBImage object

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_copy = img.copy()

        # Check that this is a new instance
        >>> id(img_copy) != id(img)
        True
        """
        # YOUR CODE GOES HERE #
        return RGBImage(self.get_pixels())

    def get_pixel(self, row, col):
        """
        Returns the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid index
        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError

        >>> img.get_pixel(0, 1)
        (0, 0, 0)

        >>> img.get_pixel(0, 2)
        Traceback (most recent call last):
        ...
        ValueError

        >>> img.get_pixel('a', 0)
        Traceback (most recent call last):
        ...
        TypeError

        >>> img.get_pixel(0, 'b')
        Traceback (most recent call last):
        ...
        TypeError

        # Run and check the returned value
        >>> img.get_pixel(0, 0)
        (255, 255, 255)
        """
        # YOUR CODE GOES HERE #
        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError()
        if row not in range(self.num_rows):
            raise ValueError()
        if col not in range(self.num_cols):
            raise ValueError()

        return tuple(self.pixels[row][col])

    def set_pixel(self, row, col, new_color):
        """
        Sets the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid new_color tuple
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        >>> img.set_pixel('a', 0, (255, 0, 0))
        Traceback (most recent call last):
        ...
        TypeError

        >>> img.set_pixel(0, 'b', (255, 0, 0))
        Traceback (most recent call last):
        ...
        TypeError

        >>> img.set_pixel(3, 0, (255, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        >>> img.set_pixel(0, 3, (255, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        >>> img.set_pixel(0, 'b', [255, 0, 0])
        Traceback (most recent call last):
        ...
        TypeError

        >>> img.set_pixel(0, 'b', (255, 0, 0, 4))
        Traceback (most recent call last):
        ...
        TypeError

        >>> img.set_pixel(0, 'b', ('255', 0, 0))
        Traceback (most recent call last):
        ...
        TypeError


        # Check that the R/G/B value with negative is unchanged
        >>> img.set_pixel(0, 0, (-1, 0, 0))
        >>> img.pixels
        [[[255, 0, 0], [0, 0, 0]]]
        """
        # YOUR CODE GOES HERE #

        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError()
        if row not in range(self.num_rows):
            raise ValueError()
        if col not in range(self.num_cols):
            raise ValueError()
        if not isinstance(new_color, tuple) or len(new_color) != 3:
            raise TypeError()
        if not all([isinstance(color, int) for color in new_color]):
            raise TypeError()
        if not all([color <= 255 for color in new_color]):
            raise ValueError()

        for i in range(len(new_color)):
            if new_color[i] < 0:
                continue
            self.pixels[row][col][i] = new_color[i]



# Part 2: Image Processing Template Methods #
class ImageProcessingTemplate:
    """
    Contains assorted image processing methods
    Intended to be used as a parent class
    """

    def __init__(self):
        """
        Creates a new ImageProcessingTemplate object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        # YOUR CODE GOES HERE #
        self.cost = 0

    def get_cost(self):
        """
        Returns the current total incurred cost

        # Check that the cost value is returned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost = 50 # Manually modify cost
        >>> img_proc.get_cost()
        50
        """
        # YOUR CODE GOES HERE #
        return self.cost

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check if this is returning a new RGBImage instance
        >>> img_proc = ImageProcessingTemplate()
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_negate = img_proc.negate(img)
        >>> id(img) != id(img_negate) # Check for new RGBImage instance
        True

        # The following is a description of how this test works
        # 1 Create a processor
        # 2/3 Read in the input and expected output
        # 4 Modify the input
        # 5 Compare the modified and expected
        # 6 Write the output to file
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()                            # 1
        >>> img = img_read_helper('img/test_image_32x32.png')                 # 2
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')  # 3
        >>> img_negate = img_proc.negate(img)                               # 4
        >>> img_negate.pixels == img_exp.pixels # Check negate output       # 5
        True
        >>> img_save_helper('img/out/test_image_32x32_negate.png', img_negate)# 6
        """
        # YOUR CODE GOES HERE #

        new_pixels = [[[255 - color for color in pixel] for pixel in row] for row in image.get_pixels()]
        return RGBImage(new_pixels)

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_gray.png')
        >>> img_gray = img_proc.grayscale(img)
        >>> img_gray.pixels == img_exp.pixels # Check grayscale output
        True
        >>> img_save_helper('img/out/test_image_32x32_gray.png', img_gray)
        """
        # YOUR CODE GOES HERE #
        new_pixels = [[[sum(pixel) // 3 for color in pixel] for pixel in row] for row in image.get_pixels()]
        return RGBImage(new_pixels)

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_rotate.png')
        >>> img_rotate = img_proc.rotate_180(img)
        >>> img_rotate.pixels == img_exp.pixels # Check rotate_180 output
        True
        >>> img_save_helper('img/out/test_image_32x32_rotate.png', img_rotate)
        """
        # YOUR CODE GOES HERE #
        pixels = image.get_pixels()
        rotated_image_pixels = [row[::-1] for row in pixels[::-1]]
        return RGBImage(rotated_image_pixels)

    def get_average_brightness(self, image):
        """
        Returns the average brightness for the given image

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.get_average_brightness(img)
        86
        """
        # YOUR CODE GOES HERE #
        return sum([sum([sum(pixel) // 3 for pixel in row]) for row in image.pixels]) // (image.num_rows * image.num_cols)

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_adjusted.png')
        >>> img_adjust = img_proc.adjust_brightness(img, 75)
        >>> img_adjust.pixels == img_exp.pixels # Check adjust_brightness
        True
        >>> img_save_helper('img/out/test_image_32x32_adjusted.png', img_adjust)

        >>> img_proc.adjust_brightness(img, 'a')
        Traceback (most recent call last):
        ...
        TypeError

        >>> img_proc.adjust_brightness(img, -257)
        Traceback (most recent call last):
        ...
        ValueError

        >>> img_proc.adjust_brightness(img, 257)
        Traceback (most recent call last):
        ...
        ValueError

        """
        # YOUR CODE GOES HERE #
        if not isinstance(intensity, int):
            raise TypeError()
        if not -255 <= intensity <= 255:
            raise ValueError()

        if intensity >= 0:
            return RGBImage([[[min(255, color + intensity) for color in pixel] for pixel in row] for row in image.get_pixels()])
        else:
            return RGBImage([[[max(0, color + intensity) for color in pixel] for pixel in row] for row in image.get_pixels()])
        

    def blur(self, image):
        """
        Returns a new image with the pixels blurred

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_blur.png')
        >>> img_blur = img_proc.blur(img)
        >>> img_blur.pixels == img_exp.pixels # Check blur
        True
        >>> img_save_helper('img/out/test_image_32x32_blur.png', img_blur)
        """
        # YOUR CODE GOES HERE #

        pixels = image.get_pixels()
        blurred = image.get_pixels()
        avg_list = lambda lst: sum(lst) // len(lst)

        for row_no in range(image.num_rows):
            for col_no in range(image.num_cols):
                r_values = []
                g_values = []
                b_values = []

                for i in range(row_no - 1, row_no + 2):
                    for j in range(col_no - 1, col_no + 2):
                        if 0 <= i < image.num_rows and 0 <= j < image.num_rows:
                            r_values.append(pixels[i][j][0])
                            g_values.append(pixels[i][j][1])
                            b_values.append(pixels[i][j][2])    
                    
                blurred[row_no][col_no] = [avg_list(r_values), avg_list(g_values), avg_list(b_values)]

        return RGBImage(blurred)


# Part 3: Standard Image Processing Methods #
class StandardImageProcessing(ImageProcessingTemplate):
    """
    Represents a standard tier of an image processor
    """

    def __init__(self):
        """
        Creates a new StandardImageProcessing object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        # YOUR CODE GOES HERE #
        self.cost = 0
        self.free_methods = 0

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check the expected cost
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> negated = img_proc.negate(img_in)
        >>> img_proc.get_cost()
        5

        # Check that negate works the same as in the parent class
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')
        >>> img_negate = img_proc.negate(img)
        >>> img_negate.pixels == img_exp.pixels # Check negate output
        True
        """
        # YOUR CODE GOES HERE #
        if self.free_methods:
            self.free_methods -= 1
        else:
            self.cost += 5
        return super().negate(image)

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        """
        # YOUR CODE GOES HERE #
        if self.free_methods:
            self.free_methods -= 1
        else:
            self.cost += 6
        
        return super().grayscale(image)

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image
        """
        # YOUR CODE GOES HERE #
        if self.free_methods:
            self.free_methods -= 1
        else:
            self.cost += 10
        return super().rotate_180(image)

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level
        """
        # YOUR CODE GOES HERE #
        if self.free_methods:
            self.free_methods -= 1
        else:
            self.cost += 1
        super().adjust_brightness(image, intensity)

    def blur(self, image):
        """
        Returns a new image with the pixels blurred
        """
        # YOUR CODE GOES HERE #
        if self.free_methods:
            self.free_methods -= 1
        else:
            self.cost += 5
        super().adjust_brightness(image)

    def redeem_coupon(self, amount):
        """
        Makes the given number of methods calls free

        # Check that the cost does not change for a call to negate
        # when a coupon is redeemed
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.redeem_coupon(1)
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        0

        >>> img_proc.redeem_coupon('a')
        Traceback (most recent call last):
        ...
        TypeError

        >>> img_proc.redeem_coupon(0)
        Traceback (most recent call last):
        ...
        ValueError
        """
        # YOUR CODE GOES HERE #
        if not isinstance(amount, int):
            raise TypeError()
        if amount <= 0:
            raise ValueError()

        self.free_methods += amount


# Part 4: Premium Image Processing Methods #
class PremiumImageProcessing(ImageProcessingTemplate):
    """
    Represents a paid tier of an image processor
    """

    def __init__(self):
        """
        Creates a new PremiumImageProcessing object

        # Check the expected cost
        >>> img_proc = PremiumImageProcessing()
        >>> img_proc.get_cost()
        50
        """
        # YOUR CODE GOES HERE #
        self.cost = 50

    def chroma_key(self, chroma_image, background_image, color):
        """
        Returns a copy of the chroma image where all pixels with the given
        color are replaced with the background image.

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> img_in_back = img_read_helper('img/test_image_32x32.png')
        >>> color = (255, 255, 255)
        >>> img_exp = img_read_helper('img/exp/square_32x32_chroma.png')
        >>> img_chroma = img_proc.chroma_key(img_in, img_in_back, color)
        >>> img_chroma.pixels == img_exp.pixels # Check chroma_key output
        True
        >>> img_save_helper('img/out/square_32x32_chroma.png', img_chroma)

        >>> img_proc.chroma_key('a', img_in, color)
        Traceback (most recent call last):
        ...
        TypeError

        >>> img_proc.chroma_key(img_in, 'b', color)
        Traceback (most recent call last):
        ...
        TypeError


        """
        # YOUR CODE GOES HERE #
        if not isinstance(chroma_image, RGBImage):
            raise TypeError()
        if not isinstance(background_image, RGBImage):
            raise TypeError()
        if chroma_image.num_rows != background_image.num_rows:
            raise ValueError()
        if chroma_image.num_cols != background_image.num_cols:
            raise ValueError()
 
        new_image = chroma_image.copy()

        for i in range(chroma_image.num_rows):
            for j in range(chroma_image.num_cols):
                if new_image.get_pixel(i, j) == color:
                    new_image.set_pixel(i, j, background_image.get_pixel(i, j))

        return new_image

    def sticker(self, sticker_image, background_image, x_pos, y_pos):
        """
        Returns a copy of the background image where the sticker image is
        placed at the given x and y position.

        # Test with out-of-bounds image and position size
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/test_image_32x32.png')
        >>> x, y = (31, 0)
        >>> img_proc.sticker(img_sticker, img_back, x, y)
        Traceback (most recent call last):
        ...
        ValueError

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/test_image_32x32.png')
        >>> x, y = (3, 3)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_sticker.png')
        >>> img_combined = img_proc.sticker(img_sticker, img_back, x, y)
        >>> img_combined.pixels == img_exp.pixels # Check sticker output
        True
        >>> img_save_helper('img/out/test_image_32x32_sticker.png', img_combined)
        """
        # YOUR CODE GOES HERE #

        if not isinstance(sticker_image, RGBImage) or not isinstance(background_image, RGBImage):
            raise TypeError()
        if sticker_image.num_rows > background_image.num_rows or sticker_image.num_cols > background_image.num_cols:
            raise ValueError()
        if not isinstance(x_pos, int) or not isinstance(y_pos, int):
            raise TypeError()
        if x_pos + sticker_image.num_cols > background_image.num_cols:
            raise ValueError()
        if y_pos + sticker_image.num_rows > background_image.num_rows:
            raise ValueError()

        new_image = background_image.copy()

        for i in range(sticker_image.num_rows):
            for j in range(sticker_image.num_cols):
                new_image.set_pixel(x_pos + i, y_pos + j, sticker_image.get_pixel(i, j))

        return new_image


    def edge_highlight(self, image):
        """
        Returns a new image with the edges highlighted

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_edge = img_proc.edge_highlight(img)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_edge.png')
        >>> img_exp.pixels == img_edge.pixels # Check edge_highlight output
        True
        >>> img_save_helper('img/out/test_image_32x32_edge.png', img_edge)
        """
        # YOUR CODE GOES HERE #

        edge_highlighted = image.copy()
        image_averaged = [[sum(pixel) // 3 for pixel in row] for row in image.get_pixels()]

        for i in range(image.num_rows):
            for j in range(image.num_cols):

                new_color = 0
                for row_no in range(i - 1, i + 2):
                    for col_no in range(j - 1, j + 2):
                        if 0 <= row_no < image.num_rows and 0 <= col_no < image.num_cols:
                            if row_no == i and col_no == j:
                                new_color += 8 * image_averaged[row_no][col_no]
                            else:
                                new_color += -1 * image_averaged[row_no][col_no]

                if new_color > 255:
                    new_color = 255
                if new_color < 0:
                    new_color = 0

                edge_highlighted.set_pixel(i, j, (new_color, new_color, new_color))

        return edge_highlighted
                            



# Part 5: Image KNN Classifier #
class ImageKNNClassifier:
    """
    Represents a simple KNNClassifier
    """

    def __init__(self, k_neighbors):
        """
        Creates a new KNN classifier object
        """
        # YOUR CODE GOES HERE #
        self.k_neighbors = k_neighbors

    def fit(self, data):
        """
        Stores the given set of data and labels for later
        """
        # YOUR CODE GOES HERE #
        if len(data) < self.k_neighbors:
            raise ValueError()

        self.data = data


    def distance(self, image1, image2):
        """
        Returns the distance between the given images

        >>> img1 = img_read_helper('img/steve.png')
        >>> img2 = img_read_helper('img/knn_test_img.png')
        >>> knn = ImageKNNClassifier(3)
        >>> knn.distance(img1, img2)
        15946.312896716909
        """
        # YOUR CODE GOES HERE #
        if not (isinstance(image1, RGBImage) and isinstance(image2, RGBImage)):
            raise TypeError()
        if not (image1.num_rows == image2.num_rows and image1.num_cols == image2.num_cols):
            raise ValueError()

        image1_pix = image1.get_pixels()
        image2_pix = image2.get_pixels()

        distance_squared = sum([sum([sum([(image1_pix[row][col][chan] - image2_pix[row][col][chan]) ** 2 for chan in range(3)]) for col in range(image1.num_cols)]) for row in range(image1.num_rows)])
        return distance_squared ** 0.5


    def vote(self, candidates):
        """
        Returns the most frequent label in the given list

        >>> knn = ImageKNNClassifier(3)
        >>> knn.vote(['label1', 'label2', 'label2', 'label2', 'label1'])
        'label2'
        """
        # YOUR CODE GOES HERE #
        max_occurences = -1
        max_label = ''

        for label in candidates:
            if candidates.count(label) > max_occurences:
                max_occurences = candidates.count(label)
                max_label = label

        return max_label

    def predict(self, image):
        """
        Predicts the label of the given image using the labels of
        the K closest neighbors to this image

        The test for this method is located in the knn_tests method below
        """
        # YOUR CODE GOES HERE #
        if not self.data:
            raise ValueError()

        distances = [(self.distance(image, tup[0]), tup[1]) for tup in self.data]
        sorted_distances = sorted(distances, key=lambda t: t[0])
        k_nearest_images = sorted_distances[:self.k_neighbors]
        k_nearest_labels = [tup[1] for tup in k_nearest_images]

        return self.vote(k_nearest_labels)



def knn_tests(test_img_path):
    """
    Function to run knn tests

    >>> knn_tests('img/knn_test_img.png')
    'nighttime'
    """
    # Read all of the sub-folder names in the knn_data folder
    # These will be treated as labels
    path = 'knn_data'
    data = []
    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        # Ignore non-folder items
        if not os.path.isdir(label_path):
            continue
        # Read in each image in the sub-folder
        for img_file in os.listdir(label_path):
            train_img_path = os.path.join(label_path, img_file)
            img = img_read_helper(train_img_path)
            # Add the image object and the label to the dataset
            data.append((img, label))

    # Create a KNN-classifier using the dataset
    knn = ImageKNNClassifier(5)

    # Train the classifier by providing the dataset
    knn.fit(data)

    # Create an RGBImage object of the tested image
    test_img = img_read_helper(test_img_path)

    # Return the KNN's prediction
    predicted_label = knn.predict(test_img)
    return predicted_label
