import cv2
import numpy as np
import math
import os
import easygui
import random
import sys
from astar import GridWithWeights, ReconstructPath, AStarSearch


def main():
    global xCoor, yCoor, mouseBut, gridSizeChanged, objSizeChanged, thresh_walls_changed, mode, drawing, imgChanged, grid_size, obj_size, thresh_walls
    # начальный угол
#     alpha = 90
    # обороты колеса в минуту, радиус колеса, ширина оси в СМ
#     rpm,wheelRad,wheelAxLen = (15,15,50)

    while True:
        msg = ""
        title = "Параметры робота"
        fieldNames = ["Обороты колеса в минуту","Радиус колеса (см)","Ширина оси (см)","Начальный угол"]
        fieldValues = easygui.multenterbox(msg,title, fieldNames)
        try:
            rpm = int(fieldValues[0])
            wheelRad = int(fieldValues[1])
            wheelAxLen = int(fieldValues[2])
            alpha = int(fieldValues[3])
        except:
            if easygui.ccbox("Данные введены некорректно. Введите заново.", title): 
                pass
            else:
                sys.exit(0)
        else:
            break
            
    #     print(rpm,wheelRad,wheelAxLen)
       
    
    # сантиметров в пикселе
    cmInPx = 10

    # кисть для режима рисования
    brush_size = 40
    brush_color = (255,0,0)
    prevXcoor = None
    prevYcoor = None 
    
    # изображения для режима рисования
    tree = cv2.imread('tree.png',-1)
    stone = cv2.imread('stone.png',-1)
    
    grid_size = 10
    # размер агента в px
    obj_size = 10
    # агент
    agent = cv2.imread('tricycle.png',-1)
    (ah, aw) = agent.shape[:2]
    # calculate the center of the image
    center = (aw / 2, ah / 2) 
    M = cv2.getRotationMatrix2D(center, alpha, 1.0)
    agent = cv2.warpAffine(agent, M, agent.shape[:2])
    
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
    start_point_px = [-1000, -1000]
#     start_point_px = [100, 100]
    goal_point_px = [-1000, -1000]
    # и координаты курсора
    xCoor = -brush_size*2
    yCoor = -brush_size*2
    
    
    # окно информации
    infoIm = cv2.imread('info.png',0)
    cv2.namedWindow ('Info', cv2.WINDOW_AUTOSIZE)
    cv2.imshow ('Info', infoIm)
    
    
#     # окно комманд
#     cv2.namedWindow("Commands")
#     # подложка окна комманд
#     cmds_img = np.zeros((512,512,3), np.uint8)
#     # шрифт
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     bottomLeftCornerOfText = (10,500)
#     fontScale = 1
#     fontColor = (255,255,255)
#     lineType = 2

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
            start_point_px = [xCoor, yCoor]
        elif mouseBut == "R":
            goal_point_px = [xCoor, yCoor]
        if drawing:
            if brush_color == 'tree' or brush_color == 'stone':
                if xCoor < img_width-brush_size+1 and \
                   yCoor < img_height-brush_size+1 and \
                   xCoor > brush_size+1 and \
                   yCoor > brush_size+1:
                    if brush_color == 'tree': over = tree
                    elif brush_color == 'stone': over = stone
                    img_gray_pre = overlay_transparent(img_gray_pre, over, xCoor-brush_size, yCoor-brush_size,(brush_size*2,brush_size*2))
            else:
                if prevXcoor is not None and prevYcoor is not None:
                    cv2.circle(img_gray_pre,(xCoor,yCoor),brush_size,brush_color,-1)
                    cv2.line(img_gray_pre, (xCoor, yCoor), (prevXcoor, prevYcoor), brush_color, brush_size*2, -1)
        prevXcoor = xCoor
        prevYcoor = yCoor    
        
        # формирование подложки
        if gridSizeChanged or objSizeChanged or imgChanged == True or thresh_walls_changed:
            # обнуляем изображение
            img_gray = img_gray_pre.copy()
            # Чистим шум
            img_gray= cv2.medianBlur(img_gray, 5)    
            # ищем препятствия, создаём подложку
            walls, pre_img, img_erode = analyze_image(img_gray, obj_size, grid_size, thresh_walls)
        # отрисовка точек и пути
        if gridSizeChanged or objSizeChanged or mouseBut == "R" or mouseBut == "L" or imgChanged == True or thresh_walls_changed:
            # расчёт координат сетки точек старта и цели
            start_point = (start_point_px[0]//grid_size, start_point_px[1]//grid_size)
            goal_point = (goal_point_px[0]//grid_size, goal_point_px[1]//grid_size)
            # переопределение координат клика мыши в центр сектора
            start_point_px[0] = start_point[0]*grid_size+grid_size//2
            start_point_px[1] = start_point[1]*grid_size+grid_size//2
            goal_point_px[0] = goal_point[0]*grid_size+grid_size//2
            goal_point_px[1] = goal_point[1]*grid_size+grid_size//2
            # отрисовка на подложке
            img = pre_img.copy()
            cv2.circle(img,(start_point_px[0],start_point_px[1]), obj_size, (190,0,0), 2)
            cv2.circle(img,(goal_point_px[0],goal_point_px[1]), obj_size, (0,0,190), 2)
            
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
                    # оптимизируем путь
                    pathProcessing(p, grid_size, img_erode)
                    # отрисовываем путь
                    for index in range(0, len(p)-1):
                        cv2.line(img, (p[index][0]*grid_size+grid_size//2, p[index][1]*grid_size+grid_size//2), (p[index+1][0]*grid_size+grid_size//2, p[index+1][1]*grid_size+grid_size//2), (0, 190, 0), obj_size, 1)
                        cv2.line(img, (p[index][0]*grid_size+grid_size//2, p[index][1]*grid_size+grid_size//2), (p[index+1][0]*grid_size+grid_size//2, p[index+1][1]*grid_size+grid_size//2), (0, 207, 196), 2, 1)
                    moveCmdsToTxt, moveCmds = moveCmdsCompil(p,alpha,grid_size,cmInPx)
                    np.savetxt('cmds_agent.txt', moveCmdsToTxt, delimiter=';',fmt = '%s')
                    motorCmds = motorCmdsCompil(moveCmds,rpm,wheelRad,wheelAxLen,cmInPx)
                    np.savetxt('cmds_motors.txt', motorCmds, delimiter=';',fmt = '%s')
                except Exception as e:
                    print("невозможно проложить путь \n \r" + str(e))
            
            if start_point_px[0] < img_width-obj_size+1 and \
                   start_point_px[1] < img_height-obj_size+1 and \
                   start_point_px[0] > obj_size+1 and \
                   start_point_px[1] > obj_size+1: 
                img = overlay_transparent(img, agent, start_point_px[0]-obj_size, start_point_px[1]-obj_size,(obj_size*2,obj_size*2))
            
                    
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
            bgDrawingImg = img_gray_pre.copy()
            if brush_color == (255,0,0):
                cv2.circle(img_gray_pre,(xCoor,yCoor),brush_size,brush_color,-1)
            elif brush_color == (255,255,255):
                cv2.circle(img_gray_pre,(xCoor,yCoor),brush_size,brush_color,-1)
                cv2.circle(img_gray_pre,(xCoor,yCoor),brush_size,(0,0,0),1)
            elif brush_color == 'tree' or brush_color == 'stone':
                if xCoor < img_width-brush_size+1 and \
                   yCoor < img_height-brush_size+1 and \
                   xCoor > brush_size+1 and \
                   yCoor > brush_size+1:
                    if brush_color == 'tree': over = tree
                    elif brush_color == 'stone': over = stone
                    img_gray_pre = overlay_transparent(img_gray_pre, over, xCoor-brush_size, yCoor-brush_size,(brush_size*2,brush_size*2))
                    drawing = False
            cv2.imshow('A-Star Pathfinding',img_gray_pre)
            img_gray_pre = bgDrawingImg
            
        
        key = cv2.waitKey(1) & 0xFF
        # смена режима
        if key == ord('m'):
            if mode == False:
                imgChanged = True
            mode = not mode
        # вывод команд для агента
        elif key == 32:
            script_dir = os.path.dirname(__file__)
            rel_path = "cmds_agent.txt"
            abs_file_path = os.path.join(script_dir, rel_path)
            os.startfile(abs_file_path)
        # вывод команд для двигателей
        elif key == 9:
            script_dir = os.path.dirname(__file__)
            rel_path = "cmds_motors.txt"
            abs_file_path = os.path.join(script_dir, rel_path)
            os.startfile(abs_file_path)
        # Esc - выход
        elif key == 27:
            break

        if not mode:
            # очистка поля для рисования
            if key == ord('c'):
                img_gray_pre[:,:] = 255
            # увеличение размера кисти
            elif key == ord('='):
                brush_size += 1
                if brush_size >= 100:
                    brush_size = 100
            # уменьшение размера кисти
            elif key == ord('-'):
                brush_size -= 1
                if brush_size <= 1:
                    brush_size = 1
            # выбор цвета кисти
            elif key == ord('w'):
                brush_color = (255,255,255)
            elif key == ord('b'):
                brush_color = (255,0,0)
            elif key == ord('t'):
                brush_color = 'tree'
            elif key == ord('s'):
                brush_color = 'stone'
            # наложение шума
            elif key == ord('n'):
                img_gray_pre = sp_noise(img_gray_pre,0.05)
                

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
    img_gray_erode = img_map.copy()
    # отображаем фантом безопасного расстояния от препятствий
    img = cv2.addWeighted(img_map,0.9,img_gray_pre,0.5,0)
    img = cv2.addWeighted(img_gray,0.9,img,0.8,0)
    # наносим сетку на изображение
    for x in range(0,img.shape[1],grid_size):
        cv2.line(img, (x, 0), (x, img.shape[0]), (255, 123, 123), 1, 1)
    for y in range(0,img.shape[0],grid_size):
        cv2.line(img, (0, y), (img.shape[1], y), (255, 123, 123), 1, 1)
    
    return walls, img, img_gray_erode


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
# постобработка траектории пути
def pathProcessing(positions, gridSize, img):
    i = 0
    while len(positions)-1 >= i+2:
        if tryGo(img, positions[i][1]*gridSize+gridSize//2, positions[i][0]*gridSize+gridSize//2, positions[i+2][1]*gridSize+gridSize//2, positions[i+2][0]*gridSize+gridSize//2):
            del(positions[i+1])
        else:
            i += 1
    
def tryGo(img, x1=0, y1=0, x2=0, y2=0):
    passable = True

    dx = x2 - x1
    dy = y2 - y1
    
    sign_x = 1 if dx>0 else -1 if dx<0 else 0
    sign_y = 1 if dy>0 else -1 if dy<0 else 0
    
    if dx < 0: dx = -dx
    if dy < 0: dy = -dy
    
    if dx > dy:
        pdx, pdy = sign_x, 0
        es, el = dy, dx
    else:
        pdx, pdy = 0, sign_y
        es, el = dx, dy
    
    x, y = x1, y1
    error, t = el/2, 0        
    
#     getPixel(x, y, (0,0,255), img_def)
    
    while t < el:
        error -= es
        if error < 0:
            error += el
            x += sign_x
            y += sign_y
        else:
            x += pdx
            y += pdy
        t += 1
        if any(t != 255 for t in img[x,y]):
            passable = False
            return passable
    return passable

# cmdsParams is (angle, dirA, dist, newAngle)
#--------------------------------------------------------------------------
# формирование команд двигателям
def motorCmdsCompil(cmdsParams,rpm,wheelRad,wheelAxLen,pxInCm):
    # рассчитываем линейную скорость перемещения см/сек
    surfSpeed = wheelRad * 2 * math.pi * rpm / 60
    cmds = []
    for c in cmdsParams:  
        # команда на вращение
        if c[0] != 0: 
            # рассчитываем длину дуги сектора поворота
            # делаем время положительным
            if c[0] < 0: sign = -1
            else: sign = 1
            arcL = math.pi * wheelAxLen/2 * sign* c[0] / 180
            # рассчитываем время на которое нужно 
            # включить оба двигателя для поворота
            workTime = arcL/surfSpeed/2
            if c[1] > 0:
                dirR = ("обратное","прямое  ")
            elif c[1] < 0 or c[1] == 0:
                dirR = ("прямое  ","обратное")
            rotateCmd = "ЛЕВЫЙ: " + dirR[0] + " ПРАВЫЙ: " + dirR[1] + " на " + '{:5.2f}'.format(workTime) + " сек."
            cmds.append(rotateCmd)
        # команда на движение вперёд
        if c[2] != 0:
            workTime = c[2] / surfSpeed / 2
            moveCmd = "ЛЕВЫЙ: прямое  " + " ПРАВЫЙ: прямое   на " + '{:5.2f}'.format(workTime) + " сек."
            cmds.append(moveCmd)
    return cmds
        
# формирование команд агенту
def moveCmdsCompil(pathCoors,alpha,grid_size, cmInPx):
    cmds = []
    cmdsParams = []
    for i in range(len(pathCoors)-1):
        (x1,y1) = pathCoors[i]
        (x2,y2) = pathCoors[i+1]
        angle, dirA, dist, newAngle = turnCmd(alpha, x1*grid_size+grid_size//2, y1*grid_size+grid_size//2, x2*grid_size+grid_size//2, y2*grid_size+grid_size//2)
        if dirA == -1: dirAs = " (по ЧС)    "
        if dirA == 1: dirAs = " (против ЧС)"
        if dirA == 0: dirAs = "clockwise"
        alpha = newAngle
        strCmd = "Повернуть на " + '{:4.0f}'.format(angle) + " град." + dirAs + " и проехать " + '{:5.0f}'.format(dist*cmInPx) +" cм. напр.: = " + '{:3.0f}'.format(newAngle)
        cmds.append(strCmd)
        cmdsParams.append((angle, dirA, dist*cmInPx, newAngle))
    return cmds, cmdsParams
        
# параметры команды на поворот - угол, знак поворота
# -1 по ЧС
# 1 - против
def turnCmd(alpha, Ax, Ay, Cx, Cy):
    # адаптируем координаты относительно матрицы изображения
#     alpha = alpha   # угол против ЧС относительно оси Х
    Ay = -Ay
    Cy = -Cy
    # находим координаты вектора направления робота
    Bx = Ax + math.cos(math.radians(alpha))
    By = Ay + math.sin(math.radians(alpha))
    # находим знак векторного произведения вектора направления робота и вектора к след. точке пути
    crossSign = np.sign((Bx-Ax) * (Cy-Ay) - (By-Ay) * (Cx-Ax))
    # находим угол между вектором направления робота и вектора к след. точке пути
    ABx = Bx-Ax
    ABy = By-Ay
    ACx = Cx-Ax
    ACy = Cy-Ay
    angle = math.degrees(math.acos((ABx*ACx + ABy*ACy) / (math.sqrt(ABx*ABx + ABy*ABy) * math.sqrt(ACx*ACx + ACy*ACy))))
    # находим новый угол направления агента
    if crossSign == -1: angle = -angle
    newAlpha = alpha + angle
#     if newAlpha > 360: newAlpha = newAlpha - 360
#     if newAlpha < 0: newAlpha = 360 - newAlpha
    # находим расстояние, которое нужно будет пройти
    distance = math.sqrt(ACx*ACx + ACy*ACy)
    return angle, crossSign, distance, newAlpha

#--------------------------------------------------------------------------
# наложение изображения с альфа-каналом
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    bg_img = background_img.copy()
    
    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    b,g,r,a = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b,g,r))
    
    mask = cv2.medianBlur(a,5)

    h, w, _ = overlay_color.shape
    roi = bg_img[y:y+h, x:x+w]

    img1_bg = cv2.bitwise_and(roi.copy(),roi.copy(),mask = cv2.bitwise_not(mask))
    
    img2_fg = cv2.bitwise_and(overlay_color,overlay_color,mask = mask)

    bg_img[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)

    return bg_img

# наложение шума сольПерец
def sp_noise(image,prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
#--------------------------------------------------------------------------
if __name__ == "__main__":
    main()


            






