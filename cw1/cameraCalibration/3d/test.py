import cv2

img = cv2.imread("data/test2.jpg")
mask_img = cv2.inRange(img, (0, 100, 0), (100, 255, 100))
conts = cv2.findContours(mask_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
# print(conts)
# print(cv2.contourArea(conts[0]))
# for item in conts:
#     print(cv2.contourArea(item))
list.sort(list(conts), key=lambda x: cv2.contourArea(x), reverse=True)
conts = conts[-6:]
print(conts)
for c in conts:
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
    # cv2.putText(
    #     img,
    #     f"({cX},{cY})",
    #     (cX + 10, cY + 10),
    #     cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    #     1,
    #     (0, 0, 0),
    # )
# cv2.drawContours(img, conts, -1, (0, 0, 0))
cv2.imwrite("output.jpg", img)
cv2.imwrite("mask.jpg", mask_img)