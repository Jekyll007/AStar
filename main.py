import cv2
import numpy as np
from astar import GridWithWeights, ReconstructPath, AStarSearch


def main():
    global xCoor, yCoor, mouseBut, gridSizeChanged, objSizeChanged, thresh_walls_changed, mode, drawing, imgChanged, grid_size, obj_size, thresh_walls

    # кисть для режима рисования
    brush_size = 10
    brush_color = (0,0,0)
    
    grid_size = 10
    obj_size = 9
    # порог обнаружения препятствий
    thresh_walls = 130
    
    # флаги для отслеживания событий
    gridSizeChanged = True
    objSizeChanged = True
    mode = True
    drawing = False
    mouseBut = ""
    # возврат из режима рисования
    imgChanged = False
    
    # графическое окно
    cv2.namedWindow("A-Star Pathfinding")
    cv2.setMouseCallback("A-Star Pathfinding", mouse_handler)
    cv2.createTrackbar( 'Grid', 'A-Star Pathfinding', grid_size, 50, gridSizeBarHandler)
    cv2.createTrackbar( 'Object', 'A-Star Pathfinding', obj_size, 50, objSizeBarHandler)
    cv2.createTrackbar( 'Thresh', 'A-Star Pathfinding', thresh_walls, 255, threshTrackbarHandler)
    # уводим точки старта и цели за границы изображения
    start_point_px = (-1000, -1000)
    goal_point_px = (-1000, -1000)

    # загрузка изображения
    img_path = 'maze.png'
    img_def = cv2.imread(img_path,1)
    if img_def is None:
        img_def = np.ones((400,800,3),np.uint8)*255

    # конвертация в бинарное
    img_gray_pre = cv2.threshold(img_def, 127, 255, cv2.THRESH_BINARY)[1]
    img_height = img_gray_pre.shape[0]
    img_width = img_gray_pre.shape[1]
     
    # цикл работы программы
    while cv2.getWindowProperty('A-Star Pathfinding',1) != -1:
        if mouseBut == "L":
            start_point_px = (xCoor, yCoor)
        elif mouseBut == "R":
            goal_point_px = (xCoor, yCoor)
        if drawing:
            cv2.circle(img_gray_pre,(xCoor,yCoor),brush_size,brush_color,-1)
        
        # формирование подложки
        if gridSizeChanged or objSizeChanged or imgChanged == True or thresh_walls_changed:
            # обнуляем изображение
            img_gray = img_gray_pre.copy()       
            # ищем препятствия, создаём подложку
            walls, pre_img = analyze_image(img_gray, obj_size, grid_size, thresh_walls)
            
        # отрисовка точек и пути
        if gridSizeChanged or objSizeChanged or mouseBut == "R" or mouseBut == "L" or imgChanged == True or thresh_walls_changed:
            # расчёт координат сетки точек старта и цели
            start_point = (start_point_px[0]//grid_size, start_point_px[1]//grid_size)
            goal_point = (goal_point_px[0]//grid_size, goal_point_px[1]//grid_size)
            # отрисовка на подложке
            img = pre_img.copy()
            cv2.circle(img,(start_point[0]*grid_size+grid_size//2,start_point[1]*grid_size+grid_size//2), obj_size, (190,0,0), -1)
            cv2.circle(img,(goal_point[0]*grid_size+grid_size//2,goal_point[1]*grid_size+grid_size//2), obj_size, (0,0,190), -1)
            
            # Расчёт пути
            if start_point != goal_point and \
               start_point not in walls and \
               goal_point not in walls and \
               start_point[0] >= 0 and start_point[1] >= 0 and goal_point[0] >= 0 and goal_point[1] >= 0:
                # создаём объект - сетку с весами
                g = GridWithWeights(img_width//grid_size+1, img_height//grid_size+1)
                # указываем список найденных препятствий
                g.walls = walls
                # запускаем поиск A*
                came_from = AStarSearch(g, start_point, goal_point)[0]
                # получаем координаты пути
                try:
                    p = ReconstructPath(came_from, start_point, goal_point)
                    # отрисовываем путь
                    for index in range(0, len(p)-1):
                        cv2.line(img, (p[index][0]*grid_size+grid_size//2, p[index][1]*grid_size+grid_size//2), (p[index+1][0]*grid_size+grid_size//2, p[index+1][1]*grid_size+grid_size//2), (0, 190, 0), 5, 1)
                except Exception:
                    pass
            
            # сброс событийных флагов
            gridSizeChanged = False
            objSizeChanged = False
            mouseBut = ""
            imgChanged = False 
            thresh_walls_changed = False
        
        # отображение сгенерированного изображения
        if mode:
            cv2.imshow('A-Star Pathfinding',img)
        else:
            cv2.imshow('A-Star Pathfinding',img_gray_pre)
        
        key = cv2.waitKey(1) & 0xFF
        # смена режима
        if key == ord('m'):
            if mode == False:
                imgChanged = True
            mode = not mode
        # Esc - выход
        elif key == 27:
            break
        # очистка поля для рисования
        if not mode:
            if key == ord('c'):
                img_gray_pre[:,:] = 255
            # увеличение размера кисти
            elif key == ord('='):
                brush_size += 1
                if brush_size >= 50:
                    brush_size = 50
            # уменьшение размера кисти
            elif key == ord('-'):
                brush_size -= 1
                if brush_size <= 1:
                    brush_size = 1
            # выбор цвета кисти
            elif key == ord('w'):
                brush_color = (255,255,255)
            elif key == ord('b'):
                brush_color = (0,0,0)

    cv2.destroyAllWindows()
    
    
#-------------------------------------------------------------------------
# ОБРАБОТЧИКИ СОБЫТИЙ
# обработчик кликов мыши
def mouse_handler(event, x, y, flags, param):
    global xCoor, yCoor, mouseBut, mode, drawing, brush_size, brush_color
    xCoor = x
    yCoor = y
    # режим поиска
    if mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            mouseBut = "L"
        elif event == cv2.EVENT_RBUTTONDOWN:
            mouseBut = "R"
    # режим рисования
    else:
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False    
        

# обработчики скроллбаров
def gridSizeBarHandler(trackbarValue):
    global gridSizeChanged, grid_size
    if trackbarValue < 5:
        trackbarValue = 5
    if trackbarValue != grid_size:
        gridSizeChanged = True
        grid_size = trackbarValue

def objSizeBarHandler(trackbarValue):
    global objSizeChanged, obj_size
    objSizeChanged = True
    obj_size = trackbarValue
    
def threshTrackbarHandler(trackbarValue):
    global thresh_walls_changed, thresh_walls
    thresh_walls_changed = True
    thresh_walls = trackbarValue
#-------------------------------------------------------------------------
# ПОЛЬЗОВАТЕЛЬСКИЕ ФУНКЦИИ
# поиск препятствий, формирование изображения для отображения
def analyze_image(img_gray,obj_size,grid_size,thresh_walls):
    img_gray_pre = img_gray.copy()
    img_height = img_gray_pre.shape[0]
    img_width = img_gray_pre.shape[1]
    # добавление отступов
    padding_size = obj_size*2   # размер отступов
    kernel = np.ones((padding_size,padding_size), np.uint8)
    img_gray = cv2.erode(img_gray, kernel, iterations = 1)
    # добавление отступов на границе
    img_gray[0:img_height,0:obj_size] = 0
    img_gray[0:obj_size,0:img_width] = 0
    img_gray[img_height-obj_size:img_height,0:img_width] = 0
    img_gray[0:img_height,img_width-obj_size:img_width] = 0
            
    # поиск препятствий
    walls = find_walls(img_gray, grid_size, thresh_walls)
    # формирование карты препятствий для отображения
    img_map = np.ones((img_height,img_width,3),np.uint8)*255
    for item in walls:
        img_map[item[1]*grid_size:item[1]*grid_size+grid_size, item[0]*grid_size:item[0]*grid_size+grid_size] = 0
    
    # отображаем фантом безопасного расстояния от препятствий
    img = cv2.addWeighted(img_map,0.9,img_gray_pre,0.5,0)
    img = cv2.addWeighted(img_gray,0.9,img,0.8,0)
    # наносим сетку на изображение
    for x in range(0,img.shape[1],grid_size):
        cv2.line(img, (x, 0), (x, img.shape[0]), (255, 123, 123), 1, 1)
    for y in range(0,img.shape[0],grid_size):
        cv2.line(img, (0, y), (img.shape[1], y), (255, 123, 123), 1, 1)
    
    return walls, img


# поочерёдно возвращает сегменты сетки и их координаты в сетке
def RetGridFrame(img, grid_size): 
    xi = 0
    for x in range(0, img.shape[1], grid_size):
        yi = 0
        for y in range(0, img.shape[0], grid_size):
            yield (xi, yi, img[y:y + grid_size, x:x + grid_size])
            yi += 1
        xi += 1


# поиск препятствий 
def find_walls(img_gray, grid_size, thresh_walls):
    walls = []  # массив с препятствиями
    # анализируем каждую ячейку сетки
    for (xi, yi, frame) in RetGridFrame(img_gray,grid_size):
#         определяем "средний" цвет сегмента сетки
#         average_color_per_row = np.average(frame, axis=0)
#         average_color = np.average(average_color_per_row, axis=0)
#         average_color = np.uint8(average_color)
        if (frame.mean() < thresh_walls):   # если сегмент достаточно чёрный
            walls.append((xi,yi))   # считаем его препятствием
    return walls
#--------------------------------------------------------------------------
if __name__ == "__main__":
    main()


            






