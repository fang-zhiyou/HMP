# 1' = 111000km
# 0.00001 = 1.11000 m
# 0.0001 = 11.1 m
# 0.001 = 111 m
# 0.2168660000000031
# 0.2834399999999988
# 2 ** 23 = 8388608
# 2 ** 22 = 4194304
# 2 ** 9 = 512  阶数为9
# 左下角 39.862142, 116.200058
# 右上角 40.079008, 116.483498
minLat = 39.862142
maxLat = 40.079008 + 0.000001
minLng = 116.200058
maxLng = 116.483498 + 0.000001

# print(maxLat - minLat)
# print(maxLng - minLng)


# 512 * 512
def get_id(lat, lng):
    step_y = (maxLat - minLat) / 512
    y = int((lat - minLat) / step_y)
    step_x = (maxLng - minLng) / 512
    x = int((lng - minLng) / step_x)
    # print(x, y)
    id = xy2d(512, x, y)
    return id


def xy2d(n, x, y):
    rx = 0
    ry = 0
    s = int(n / 2)
    d = 0
    while s > 0:
        rx = (x & s) > 0
        ry = (y & s) > 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = rot(n, x, y, rx, ry)
        s = int(s / 2)
    return d


def rot(n, x, y, rx, ry):
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        t = x
        x = y
        y = t
    return x, y
