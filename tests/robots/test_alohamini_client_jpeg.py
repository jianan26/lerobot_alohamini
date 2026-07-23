import json

import cv2
import numpy as np

from lerobot.robots.alohamini.alohamini_client import AlohaMiniClient


def test_multipart_observation_caches_original_jpeg_bytes():
    client = object.__new__(AlohaMiniClient)
    client._cameras_ft = {"forward": (8, 8, 3)}
    client._state_order = ()
    client.logs = {}
    client.last_frames = {}
    client.last_jpeg_images = {}
    client.last_remote_state = {}
    client._observation_sequence = 0

    source_frame = np.full((8, 8, 3), (10, 80, 160), dtype=np.uint8)
    encoded, jpeg = cv2.imencode(".jpg", source_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    assert encoded
    jpeg_bytes = jpeg.tobytes()
    client._poll_and_get_latest_message = lambda: [
        json.dumps({}).encode(),
        b"forward",
        jpeg_bytes,
    ]

    frames, _state = client._get_data()

    expected_frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(frames["forward"], expected_frame)
    assert client.get_latest_jpeg_images() == {"forward": jpeg_bytes}
    assert client.get_latest_jpeg_images()["forward"] is jpeg_bytes
