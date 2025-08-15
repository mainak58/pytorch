import cv2

path_name = input("Please give me your file path name : ")

if not path_name:
    print("please enter a valid path name : ")

else:
    print("\nChoose what you want to draw:")
    print("1 = Line")
    print("2 = Rectangle")
    print("3 = Circle")

    choice = input("Enter your choice (1/2/3): ")

    read_image = cv2.imread(path_name)

    match choice:
        case "1":
            print("drawing line")
            drawing_image = cv2.line(read_image, (50, 50), (400, 400), (0, 0, 255), 5)

        case "2":
            print("drawing rectangle")
            top_left = (100, 100)
            bottom_right = (400, 300)
            drawing_image = cv2.rectangle(read_image, top_left, bottom_right, (0, 0, 255), 5)

        case "3":
            print("drawing circle")
            center = (250, 250)
            radius = 100
            drawing_image = cv2.circle(read_image, center, radius, (0, 0, 255), 5)

        case _:
            print("Invalid choice!")
            exit()

    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Result", 500, 500)
    cv2.imshow("Result", drawing_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
