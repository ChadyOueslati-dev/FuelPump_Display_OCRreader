import cv2
import numpy as np

from app.ocr.preprocess import crop, find_screen, row_crop


def photo(width=400, height=300, background=90):
    return np.full((height, width, 3), background, np.uint8)


def test_finds_bright_rectangle():
    image = photo()
    cv2.rectangle(image, (100, 50), (300, 200), (255, 240, 230), -1)
    cv2.putText(image, "50.00", (120, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (40, 40, 40), 3)

    x1, y1, x2, y2 = find_screen(image)

    # Within the padding of the true edges.
    assert abs(x1 - 100) <= 8 and abs(y1 - 50) <= 8
    assert abs(x2 - 300) <= 8 and abs(y2 - 200) <= 8


def test_finds_dark_led_panel_on_a_lighter_pump():
    image = photo(background=170)
    cv2.rectangle(image, (100, 50), (300, 200), (15, 15, 15), -1)
    cv2.putText(image, "50.00", (120, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (40, 40, 255), 3)

    x1, y1, x2, y2 = find_screen(image)

    assert abs(x1 - 100) <= 8 and abs(y1 - 50) <= 8
    assert abs(x2 - 300) <= 8 and abs(y2 - 200) <= 8


def test_dark_photo_with_only_lit_digits_has_no_screen_outline():
    image = photo(background=20)
    cv2.putText(image, "50.00", (120, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (40, 40, 255), 3)
    assert find_screen(image) is None


def test_screen_cut_off_by_the_photo_edge_is_still_found():
    image = photo()
    cv2.rectangle(image, (100, 0), (300, 150), (255, 240, 230), -1)
    box = find_screen(image)
    assert box is not None and box[1] == 0


def test_larger_non_rectangular_bright_area_is_skipped():
    image = photo()
    cv2.circle(image, (290, 150), 100, (255, 255, 255), -1)
    cv2.line(image, (290, 150), (20, 280), (255, 255, 255), 12)
    cv2.rectangle(image, (20, 20), (150, 120), (255, 240, 230), -1)
    x1, y1, x2, y2 = find_screen(image)
    assert x2 < 170 and y2 < 140


def test_uniform_photo_has_no_screen():
    assert find_screen(photo()) is None
    assert find_screen(photo(background=255)) is None


def test_small_bright_spot_is_not_a_screen():
    image = photo()
    cv2.rectangle(image, (10, 10), (40, 30), (255, 255, 255), -1)
    assert find_screen(image) is None


def test_crop():
    image = photo()
    assert crop(image, (100, 50, 300, 200)).shape == (150, 200, 3)


def test_row_crop_keeps_the_row_with_a_border():
    image = photo()
    row = row_crop(image, (100, 100, 200, 140))  # 100 x 40 text box
    # 15% wider each side, 35% taller each side, plus a 20 px border all round
    assert row.shape == (40 + 2 * 14 + 2 * 20, 100 + 2 * 15 + 2 * 20, 3)


def test_row_crop_at_the_image_edge():
    row = row_crop(photo(), (0, 0, 100, 40))
    assert row.shape[0] > 40 and row.shape[1] > 100


def test_white_screen_is_found_on_an_equally_bright_coloured_pump():
    # A vivid blue pump body is as bright as the LCD; only the LCD is white.
    image = np.zeros((300, 400, 3), np.uint8)
    image[:] = (255, 120, 20)  # blue (BGR)
    cv2.rectangle(image, (120, 80), (300, 220), (235, 240, 235), -1)
    cv2.putText(image, "25.37", (140, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (60, 60, 60), 3)

    x1, y1, x2, y2 = find_screen(image)

    assert abs(x1 - 120) <= 8 and abs(y1 - 80) <= 8
    assert abs(x2 - 300) <= 8 and abs(y2 - 220) <= 8
