import numpy as np
import cv2
from chessboard_location.chessboard_finder import get_chessboard_intersections
from piece_detection.utils_chess import create_chessboard_from_board_array
from piece_detection.utils_corners import create_chessboard_array_from_assignments, denormalize_piece_info, get_squares_from_corners, is_top_left_white, match_pieces_with_squares
from piece_detection.utils_yolo import predict_image

def return_board_from_image(img, model, log = True, isRoboflow = False):

    if log:
        print("-----------------------")
        print("YOLOv5 detecting img...")
        print("-----------------------")
    # prediction = predict_image(img, model)
    # model_output_denormalized = denormalize_piece_info(prediction, img.shape[1], img.shape[0])

    if log:
        print("-----------------------")
        print("Getting chessboard intersections...")
        print("-----------------------")
    corners = get_chessboard_intersections(img)
    if corners is None:
        return None
    
    np.save("cornersmb", corners)
    print("array corners saved to file")
    print()
    
    ########### inserted some lines  print out outer and save array of corners to file ###########
    # corners clockwise starting upper left (yx values)
    c1 = corners[0][0]
    c2 = corners[8][0]
    c3 = corners[8][8]
    c4 = corners[0][8]

    ### if needed invert corners (xy):
    c1inv = c1[::-1]
    c2inv = c2[::-1]    
    c3inv = c3[::-1]
    c4inv = c4[::-1]

    print("corner 1 y,x", c1)
    print("corner 2 y,x", c2)
    print("corner 3 y,x", c3)
    print("corner 4 y,x", c4)

    img = cv2.circle(img, (c1[1], c1[0]), radius=3, color=(0, 255, 0), thickness= -1)
    img = cv2.circle(img, (c2[1], c2[0]), radius=3, color=(0, 255, 0), thickness= -1)
    img = cv2.circle(img, (c3[1], c3[0]), radius=3, color=(0, 255, 0), thickness= -1)
    img = cv2.circle(img, (c4[1], c4[0]), radius=3, color=(0, 255, 0), thickness= -1)

    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 

    ###########
    return  ###
    ###########
    
    squares = get_squares_from_corners(corners)

    assigned_squares_list = match_pieces_with_squares(squares, model_output_denormalized)
    chessboard_array = create_chessboard_array_from_assignments(assigned_squares_list)

    fix_color = False # No need for current dataset
    if fix_color and not is_top_left_white(img, squares):
        chessboard_array = np.rot90(chessboard_array, 1, (0, 1))

    chessboard = create_chessboard_from_board_array(chessboard_array, isRoboflow)

    return chessboard
