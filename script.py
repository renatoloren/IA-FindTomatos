import numpy as np
import matplotlib.pyplot as plt
import cv2

#lista de imagens para processamento
images = [1, 2, 4, 7, 8, 9, 10]
fonte = cv2.FONT_HERSHEY_SIMPLEX


#função responsável por delimitar area dos tomates e adicionar legendas
def drawContornos(list_cortornos, min_area, color, text):

        n_tomates = 0
        
        #para cada contorno
        for c in list_cortornos:
            c_area = cv2.contourArea(c)

            #se a area do contorno for grande o suficiente
            if (c_area > min_area):
                n_tomates = n_tomates + 1
                #delimitamos um circulo como area do tomate e encontramos seu centro
                (x,y),radius = cv2.minEnclosingCircle(c)
                center = (int(x),int(y))
                
                #desenhamos a legenda com status do tomate
                pt1 = (center[0], center[1])
                pt2 = (center[0]-20, center[1]+30)
                pt3 = (center[0]+20, center[1]+30)
                triangle_cnt = np.array( [pt1, pt2, pt3] )

                cv2.drawContours(img_rgb, [triangle_cnt], 0, color, -1)
                cv2.rectangle(img_rgb, (center[0]-100, center[1]+25), (center[0]+100, center[1]+70), color=color, thickness=-1)
                cv2.putText(img_rgb, text, (center[0]-70, center[1]+55), fonte, 1, (0,0,0), 1, cv2.LINE_AA)
                
        return n_tomates

def printLog(type, numbers):
    print(f'Tomates para {type} encontrados: {numbers}')

#para cada imagem
for i in images: 
    
    img = cv2.imread(f"tom{i}.jpg")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #mascara de seleção vermelha, para selecionar os tomates em estágio de consumo
    red_hsv_lower = np.array([1, 160, 120])  
    red_hsv_upper = np.array([11, 255, 255])
    red_mask_1 = cv2.inRange(img_hsv, red_hsv_lower, red_hsv_upper)

    #afinamos a mascara para remover o máximo de falso positivos com método erode
    #ampliamos o resultado do afinamento para que os pixels restantes possam possuir uma maior área 
    #e representar um tomate para consumo
    red_mask_2 = cv2.erode(red_mask_1, None, iterations=10)
    red_mask_2 = cv2.dilate(red_mask_2, None, iterations=3)
    contornos_red, _ = cv2.findContours(red_mask_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #mascara de seleção amarela/laranja, para selecionar os tomates em estágio de colheita
    yellow_hsv_lower = np.array([12, 151, 180])  
    yellow_hsv_upper = np.array([24, 255, 255])
    yellow_mask_1 = cv2.inRange(img_hsv, yellow_hsv_lower, yellow_hsv_upper)

    #utilizamos o método erode para remover fundos e pequenos detalhes de luz e reduzir o número
    # de possíveis falsos positivos 
    yellow_mask_2 = cv2.erode(yellow_mask_1, None, iterations=5)
    contornos_yellow, _ = cv2.findContours(yellow_mask_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #delimitamos o espaço de cada tomate encontrado e exibimos
    consumo_qtd = drawContornos(contornos_red, 3500, (255, 213, 0), 'consumo')
    colheita_qtd = drawContornos(contornos_yellow, 2500, (111, 255, 0), 'colheita')

    plt.imshow(img_rgb, cmap="Greys_r", vmin=0, vmax=255)
    plt.show()

    printLog('consumo', consumo_qtd)
    printLog('colheita', colheita_qtd)
