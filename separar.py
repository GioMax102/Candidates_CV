import os
import shutil
import random
from pathlib import Path

def dividir_datos(ruta_imagenes, ruta_labels, ruta_salida, ratio_entrenamiento=0.7):
    """
    Divide imágenes y labels en conjuntos de entrenamiento y validación
    
    Args:
        ruta_imagenes (str): Ruta a la carpeta con imágenes
        ruta_labels (str): Ruta a la carpeta con labels
        ruta_salida (str): Ruta donde se crearán las carpetas de salida
        ratio_entrenamiento (float): Ratio para entrenamiento (0.0-1.0)
    """
    
    # Verificar que las carpetas de entrada existen
    if not os.path.exists(ruta_imagenes):
        print(f"❌ Error: La carpeta de imágenes '{ruta_imagenes}' no existe")
        return
    
    if not os.path.exists(ruta_labels):
        print(f"❌ Error: La carpeta de labels '{ruta_labels}' no existe")
        return
    
    # Crear estructura de carpetas de salida
    carpetas_salida = [
        os.path.join(ruta_salida, "imagenes", "train"),
        os.path.join(ruta_salida, "imagenes", "val"),
        os.path.join(ruta_salida, "labels", "train"),
        os.path.join(ruta_salida, "labels", "val")
    ]
    
    for carpeta in carpetas_salida:
        os.makedirs(carpeta, exist_ok=True)
        print(f"✅ Carpeta creada: {carpeta}")
    
    # Obtener lista de archivos de imágenes (sin extensión)
    extensiones_imagen = ['.jpg', '.jpeg', '.png', '.bmp']
    archivos_imagen = []
    
    for archivo in os.listdir(ruta_imagenes):
        if any(archivo.lower().endswith(ext) for ext in extensiones_imagen):
            nombre_base = os.path.splitext(archivo)[0]
            archivos_imagen.append(nombre_base)
    
    print(f"📊 Total de imágenes encontradas: {len(archivos_imagen)}")
    
    # Verificar que existen los labels correspondientes
    archivos_validos = []
    for archivo in archivos_imagen:
        label_path = os.path.join(ruta_labels, f"{archivo}.txt")
        if os.path.exists(label_path):
            archivos_validos.append(archivo)
        else:
            print(f"⚠️  Advertencia: No se encontró label para {archivo}")
    
    print(f"📊 Archivos válidos (con imagen y label): {len(archivos_validos)}")
    
    if len(archivos_validos) == 0:
        print("❌ Error: No se encontraron pares válidos de imagen-label")
        return
    
    # Mezclar los archivos aleatoriamente
    random.shuffle(archivos_validos)
    
    # Calcular puntos de división
    punto_division = int(len(archivos_validos) * ratio_entrenamiento)
    entrenamiento = archivos_validos[:punto_division]
    validacion = archivos_validos[punto_division:]
    
    print(f"🚀 Dividiendo datos:")
    print(f"   📁 Entrenamiento: {len(entrenamiento)} archivos ({len(entrenamiento)/len(archivos_validos)*100:.1f}%)")
    print(f"   📁 Validación: {len(validacion)} archivos ({len(validacion)/len(archivos_validos)*100:.1f}%)")
    
    # Función para copiar archivos
    def copiar_archivos(lista_archivos, tipo_conjunto):
        for archivo_base in lista_archivos:
            # Encontrar la extensión real de la imagen
            imagen_origen = None
            for ext in extensiones_imagen:
                posible_path = os.path.join(ruta_imagenes, f"{archivo_base}{ext}")
                if os.path.exists(posible_path):
                    imagen_origen = posible_path
                    break
            
            if imagen_origen is None:
                print(f"⚠️  No se pudo encontrar la imagen para {archivo_base}")
                continue
            
            # Rutas de destino
            imagen_destino = os.path.join(ruta_salida, "imagenes", tipo_conjunto, os.path.basename(imagen_origen))
            label_origen = os.path.join(ruta_labels, f"{archivo_base}.txt")
            label_destino = os.path.join(ruta_salida, "labels", tipo_conjunto, f"{archivo_base}.txt")
            
            # Copiar archivos
            try:
                shutil.copy2(imagen_origen, imagen_destino)
                shutil.copy2(label_origen, label_destino)
            except Exception as e:
                print(f"❌ Error copiando {archivo_base}: {e}")
    
    # Copiar archivos de entrenamiento
    print("📤 Copiando archivos de entrenamiento...")
    copiar_archivos(entrenamiento, "train")
    
    # Copiar archivos de validación
    print("📤 Copiando archivos de validación...")
    copiar_archivos(validacion, "val")
    
    print("✅ ¡División completada!")
    print(f"📁 Resultado en: {ruta_salida}")
    print(f"   ├── imagenes/")
    print(f"   │   ├── train/ ({len(entrenamiento)} imágenes)")
    print(f"   │   └── val/ ({len(validacion)} imágenes)")
    print(f"   └── labels/")
    print(f"       ├── train/ ({len(entrenamiento)} labels)")
    print(f"       └── val/ ({len(validacion)} labels)")

def main():
    """
    Función principal - Configura tus rutas aquí
    """
    # CONFIGURA ESTAS RUTAS SEGÚN TU CASO
    RUTA_IMAGENES = "/home/mariogiovanni//Documents/yolov11_candidates/proyecto/images"  # Cambia esta ruta
    RUTA_LABELS = "/home/mariogiovanni//Documents/yolov11_candidates/proyecto/labels"      # Cambia esta ruta
    RUTA_SALIDA = "/home/mariogiovanni//Documents/yolov11_candidates/proyecto"  # Cambia esta ruta
    RATIO_ENTRENAMIENTO = 0.7  # 70% entrenamiento, 30% validación
    
    print("🔍 Verificando configuración...")
    print(f"   📸 Imágenes: {RUTA_IMAGENES}")
    print(f"   📝 Labels: {RUTA_LABELS}")
    print(f"   📂 Salida: {RUTA_SALIDA}")
    print(f"   📊 Ratio entrenamiento: {RATIO_ENTRENAMIENTO*100}%")
    
    # Ejecutar la división
    dividir_datos(RUTA_IMAGENES, RUTA_LABELS, RUTA_SALIDA, RATIO_ENTRENAMIENTO)

if __name__ == "__main__":
    main()