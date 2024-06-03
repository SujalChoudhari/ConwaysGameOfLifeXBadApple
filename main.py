import pygame
import numpy as np
import cv2

# Constants
WIDTH, HEIGHT = 600, 600
CELL_SIZE = 2
GRID_WIDTH, GRID_HEIGHT = WIDTH // CELL_SIZE, HEIGHT // CELL_SIZE
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def create_grid(width, height):
    """
    Generate a random grid of binary values with the given width and height.

    Parameters:
        width (int): The width of the grid.
        height (int): The height of the grid.

    Returns:
        numpy.ndarray: A 2D array of binary values with the specified width and height.
    """
    return np.random.choice([0, 1], size=(width, height))


def brightness_to_state(brightness):
    """
    Get the state of the cell based on the brightness of the pixel.

    Parameters:
        brightness (int): The brightness of the pixel.

    Returns:
        int: The state of the cell. 1 if the brightness is greater than 128, 0 otherwise.
    """
    return 1 if brightness > 128 else 0

def is_new_state(grid, frame):
    """
    Get the new state of Conway's Game of Life based on the current state and the video frame.

    Parameters:
        grid (numpy.ndarray): The current state of the grid.
        frame (numpy.ndarray): The video frame.

    Returns:
        numpy.ndarray: The new state of the grid.
    """
    neighbor_kernel = np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]])

    # Count live neighbors for each cell
    live_neighbors = cv2.filter2D(grid.astype(np.uint8), -1, neighbor_kernel, borderType=cv2.BORDER_CONSTANT)

    # Incorporate video pixel brightness as neighbors
    frame_resized = cv2.resize(frame, (GRID_WIDTH, GRID_HEIGHT))
    frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2GRAY)
    pixel_neighbors = (frame_gray > 128).astype(np.uint8)  # Map pixel brightness to 0 or 1

    # Check if the frame is completely black
    black_pixel_ratio = np.mean(frame_gray < 128)
    if black_pixel_ratio >= 0.95:  # If more than 95% of pixels are black
        # Combine live neighbors with pixel neighbors
        total_neighbors = live_neighbors + pixel_neighbors

        # Apply Conway's Game of Life rules
        new_grid = np.zeros_like(grid)
        new_grid[(grid == 1) & ((total_neighbors == 2) | (total_neighbors == 3))] = 1
        new_grid[(grid == 0) & (total_neighbors == 3)] = 1

        return new_grid

    # If the frame is not completely black, make the black areas hostile
    else:

        # Combine live neighbors with pixel neighbors
        total_neighbors = live_neighbors + pixel_neighbors

        # Apply Conway's Game of Life rules to non-black areas
        new_grid = np.zeros_like(grid)
        new_grid[(grid == 1) & ((total_neighbors == 2))] = 1
        new_grid[(grid == 0) & (total_neighbors == 3)] = 1
        new_grid[(total_neighbors > 8) & (grid == 0)] = 1
        return new_grid


def save_and_preview(video_file):
    """
    Save and preview the video.

    Parameters:
        video_file (str): The path to the video file.
    """
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Conway's Game of Life with Video Preview")
    clock = pygame.time.Clock()

    # Initialize CGOL grid
    grid = create_grid(GRID_WIDTH, GRID_HEIGHT)

    # Open the video file
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Output video settings
    out_width, out_height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out_fps = cap.get(cv2.CAP_PROP_FPS)
    out_video = cv2.VideoWriter("cgol_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (int(out_width), int(out_height)))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Read the next frame from the video
        ret, frame = cap.read()
        if not ret:
            # print("End of video. Looping...")
            # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            running = False
            continue

        # Simulate Conway's Game of Life for the current frame
        for _ in range(3):  # Simulate for 5 iterations
            grid = is_new_state(grid, frame)

        # Draw the grid on the Pygame screen
        screen.fill(BLACK)
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                color = WHITE if grid[row, col] == 1 else BLACK
                pygame.draw.rect(
                    screen, color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                )

        # Display the Pygame screen
        pygame.display.flip()

        # Save the CGOL frame to the output video
        cgol_frame = np.zeros((GRID_HEIGHT, GRID_WIDTH, 3), dtype=np.uint8)
        cgol_frame[grid == 1] = [255, 255,255]  # White cells
        cgol_frame[grid == 0] = [0, 0, 0]  # Black cells
        cgol_frame_resized = cv2.resize(cgol_frame, (int(out_width), int(out_height)),interpolation=cv2.INTER_NEAREST)
        out_video.write(cgol_frame_resized)

        # Limit frames per second
        clock.tick(FPS)

    # Release video objects
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()

    print("CGOL simulation complete. Output video saved as cgol_output.mp4")
    pygame.quit()

if __name__ == "__main__":
    video_file = "source.mp4"
    save_and_preview(video_file)
