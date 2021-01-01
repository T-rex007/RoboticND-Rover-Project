import numpy as np
import cv2

# Define a function to convert from image coords to rover coords
def roverCoords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel

# Define a function to convert to radial coords in rover space
def toPolarCoords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotatePix(xpix, ypix, yaw):
    # Convert yaw to radians
    #print("xpix: {}, ypix: {}, yaw: ,yaw {}".format( xpix, ypix,yaw) )
    #print(yaw)
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translatePix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pixToWorld(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotatePix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translatePix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspectTransform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped


# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def navigableTerrainColorThresh(img, rgb_thresh=(160, 160, 160)):
    """
    Returns an Image where each pixel is either a 0 or a 1 based of whether that pixel of the input
    image is below or above a certain value
    
    Args:
        img: Input Image
        rbg_thresh: pixel Thressol
        
    """
    # Create an array of zeros same xy size as img, but single channel
    binary_color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    binary_color_select[above_thresh] = 1
    # Return the binary image
    return binary_color_select

def brickColorThresh(img, rgb_thresh=((130,200), (100,180), (0, 50))):
    """
    Returns an Image where each pixel is either a 0 or a 1 based of whether that pixel of the input
    image is below or above a certain value
    
    Args:
        img: Input Image
        rbg_thresh: pixel Thressol
        
    """
    brick_color_select = (np.where(img[:,:, 0] > rgb_thresh[0][0], 1, 0) & np.where(img[:,:, 0] < rgb_thresh[0][1], 1, 0)) \
                    & (np.where(img[:,:, 1] > rgb_thresh[1][0], 1, 0) & np.where(img[:,:, 1] < rgb_thresh[1][1], 1, 0)) \
                    & (np.where(img[:,:, 2] >= rgb_thresh[2][0], 1, 0) & np.where(img[:,:, 2] < rgb_thresh[2][1], 1, 0))
    # Return the binary image
    return brick_color_select

def obstacleTerrainColorThresh(img, rgb_thresh=(50, 50, 50)):
    """
    Returns an Image where each pixel is either a 0 or a 1 based of whether that pixel of the input
    image is below or above a certain value
    
    Args:
        img: Input Image
        rbg_thresh: pixel Thressol
        
    """
    # Create an array of zeros same xy size as img, but single channel
    binary_color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] < rgb_thresh[0]) \
                & (img[:,:,1] < rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    binary_color_select[above_thresh] = 1
    # Return the binary image
    return binary_color_select


# Apply the above functions in succession and update the Rover state accordingly
def perceptionStep(Rover):
    """
    
    """
    # Perform perception steps to update Rover()
 
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    dst_size = 5 
    bottom_offset = 6                                                                               
    src = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    dst = np.float32([[Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset], 
                  [Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
                  ])
    # 2) Apply perspective transform
    warped = perspectTransform(Rover.img, src, dst)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples\
    navigable_color_select = navigableTerrainColorThresh(warped)
    brick_color_select = brickColorThresh(warped)
    obstacle_color_select = obstacleTerrainColorThresh(warped)
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    Rover.vision_image[:,:, 0] = navigable_color_select
    Rover.vision_image[:,:, 1] = brick_color_select
    Rover.vision_image[:,:, 2] =  obstacle_color_select
    # 5) Convert map image pixel values to rover-centric coords
    xpix_navigable, ypix_navigable = roverCoords(Rover.vision_image[:,:,0])
    xpix_brick, ypix_brick = roverCoords(Rover.vision_image[:,:,1])
    xpix_obstacle, ypix_obstacle = roverCoords(Rover.vision_image[:,:,2])

    # 6) Convert rover-centric pixel values to world coordinates
    xpix_world_navigable, ypix_world_navigable = pixToWorld(xpix_navigable, ypix_navigable, xpos=Rover.pos[0], ypos=Rover.pos[1], yaw = Rover.yaw, 
                                                            world_size= Rover.worldmap.shape[0], scale = 10)
    xpix_world_brick, ypix_world_brick = pixToWorld(xpix_brick, ypix_brick, xpos=Rover.pos[0], ypos=Rover.pos[1], yaw = Rover.yaw, 
                                                            world_size= Rover.worldmap.shape[0], scale = 10)
    xpix_world_obstacle, ypix_world_obstacle = pixToWorld(xpix_obstacle, ypix_obstacle, xpos=Rover.pos[0], ypos=Rover.pos[1], yaw = Rover.yaw, 
                                                            world_size= Rover.worldmap.shape[0], scale = 10)
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1'
    
    Rover.worldmap[xpix_world_navigable, ypix_world_navigable,0] += 1
    Rover.worldmap[xpix_world_brick, ypix_world_brick,1] += 1
    Rover.worldmap[xpix_world_obstacle, ypix_world_obstacle,2] += 1
    # 8) Convert rover-centric pixel positions to polar coordinates
    r, theta = toPolarCoords(xpix_navigable, ypix_navigable)
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    Rover.nav_dist = r
    Rover.nav_angles = theta 
   
    return Rover