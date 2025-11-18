import base64
import io
import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from django.shortcuts import render
from django.conf import settings
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from urllib.parse import quote_plus

# Import your separate modules
from .New_ROI_Extraction import extract_roi_from_image_array  # Your ROI extraction code
from .particle_detection import process_image                 # Your pipeline code
from .location_locator_for_django import get_lat_lng          # Your geocoding logic

def load_environment_variables():
    """
    Load environment variables from a .env file if it exists.
    """
    BASE_DIR = Path(__file__).resolve().parent.parent
    dotenv_path = os.path.join(BASE_DIR, '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
    else:
        print(f".env file not found at {dotenv_path}")


def particle_analysis(request):
    """
    Main view for processing air pollution analysis:
    - Collect dates, location, image upload, ROI confirmation, and display results.
    """
    load_environment_variables()
    context = {}
    step = 1  # Default step

    if request.method == 'POST':

        # Step 1: Collect experiment dates
        if 'submit_dates' in request.POST:
            start_date = request.POST.get('start_date')
            end_date = request.POST.get('end_date')
            if start_date and end_date:
                if start_date > end_date:
                    context['error_message'] = 'La fecha de inicio debe ser anterior a la fecha de retirada'
                    step = 1
                else:
                    request.session['start_date'] = start_date
                    request.session['end_date'] = end_date
                    step = 2
            else:
                context['error_message'] = 'Introduce ambas fechas'
                step = 1

        # Step 2(A): Geocode location name
        elif 'location_step' in request.POST:
            location_name = request.POST.get('location_name')
            lat_lng = get_lat_lng(location_name)
            if lat_lng:
                request.session['location_name'] = location_name
                request.session['latitude'] = lat_lng['latitude']
                request.session['longitude'] = lat_lng['longitude']
                google_maps_api_key = os.environ.get('GOOGLE_MAPS_API_KEY')
                if not google_maps_api_key:
                    context['error_message'] = "Google Maps API Key is not set."
                    step = 2
                else:
                    context['location_name'] = location_name
                    context['latitude'] = lat_lng['latitude']
                    context['longitude'] = lat_lng['longitude']
                    context['google_maps_api_key'] = google_maps_api_key
                    step = 2
            else:
                context['error_message'] = 'Localización no encontrada, por favor prueba de nuevo'
                step = 2

        # Step 2(B): Confirm location
        elif 'confirm_location' in request.POST:
            final_lat = request.POST.get('final_lat')
            final_lng = request.POST.get('final_lng')
            if not final_lat or not final_lng:
                context['error_message'] = 'No se han podido confirmar las coordenadas finales, por favor prueba de nuevo'
                step = 2
            else:
                request.session['latitude'] = final_lat
                request.session['longitude'] = final_lng
                step = 3

        # Step 3: Upload & classify image
        elif 'analyze_image' in request.POST and request.FILES.get('image'):
            uploaded_file = request.FILES['image']
            max_size = 10 * 1024 * 1024  # 10MB
            if uploaded_file.size > max_size:
                context['error_message'] = "La imagen es demasiado pesada, por favor usa una menor a 10 MB."
                step = 3
            else:
                image_bytes = uploaded_file.read()
                request.session['input_image_b64'] = base64.b64encode(image_bytes).decode('utf-8')

                image_stream = io.BytesIO(image_bytes)

                # Load classification model
                model_path = 'model.h5'  # Adjust if needed
                try:
                    model = tf.keras.models.load_model(model_path)
                except Exception as e:
                    context['error_message'] = f"Error loading classification model: {e}"
                    step = 3
                    return render(request, 'particle_analysis.html', context)

                # Preprocess for classification
                img = load_img(image_stream, target_size=(150, 150))
                img_array = img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                prediction = model.predict(img_array)
                label = int(prediction.round())  # 0 or 1

                if label == 0:
                    context['error_message'] = "La imagen subida no es correcta, se espera una imagen del sensor de papel AmiAire"
                    step = 2
                else:
                    image_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                    image_with_contour, roi = extract_roi_from_image_array(image_bgr)

                    if roi is not None:
                        roi_byt = cv2.imencode('.png', roi)[1].tobytes()
                        request.session['roi_image_b64'] = base64.b64encode(roi_byt).decode('utf-8')

                        if image_with_contour is not None:
                            contour_byt = cv2.imencode('.png', image_with_contour)[1].tobytes()
                            request.session['contour_image_b64'] = base64.b64encode(contour_byt).decode('utf-8')
                        else:
                            request.session['contour_image_b64'] = None

                        step = 4
                    else:
                        context['error_message'] = "No se ha detectado la región de interés, por favor prueba de nuevo"
                        step = 2

        # Step 4: Confirm or reject ROI
        elif 'confirm_roi' in request.POST:
            roi_image_b64 = request.session.get('roi_image_b64', None)
            input_image_b64 = request.session.get('input_image_b64', None)
            if roi_image_b64 and input_image_b64:
                roi_bytes = base64.b64decode(roi_image_b64)
                np_roi = np.frombuffer(roi_bytes, np.uint8)
                roi_bgr = cv2.imdecode(np_roi, cv2.IMREAD_COLOR)

                # Process the image to get analysis results
                pipeline_results = process_image(roi_bgr)
                analysis_results = pipeline_results["analysis_results"]
                pollution_data = pipeline_results["pollution_data"]
                classification_str = pipeline_results["classification"]
                binary_b64 = pipeline_results["binary_mask_b64"]
                overlay_b64 = pipeline_results["overlay_b64"]

                context['analysis_results'] = analysis_results
                context['pollution_level'] = classification_str
                context['standard_concentration'] = pollution_data.get("concentration_standard", 0.0)
                context['binary_b64'] = binary_b64
                context['overlay_b64'] = overlay_b64

                context['roi_image_b64'] = roi_image_b64
                context['contour_image_b64'] = request.session.get('contour_image_b64', None)

                # Insert results into MongoDB
                try:
                    mongo_username = os.environ.get('MONGO_USERNAME')
                    mongo_password = os.environ.get('MONGO_PASSWORD')
                    mongo_host = os.environ.get('MONGO_HOST', 'localhost')
                    mongo_port = os.environ.get('MONGO_PORT', '27017')
                    db_name = os.environ.get('DB_NAME', 'air_quality')
                    collection_name = os.environ.get('COLLECTION_NAME', 'results')

                    mongo_username_encoded = quote_plus(mongo_username)
                    mongo_password_encoded = quote_plus(mongo_password)
                    mongo_uri = f"mongodb://{mongo_username_encoded}:{mongo_password_encoded}@{mongo_host}:{mongo_port}/"

                    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
                    client.admin.command('ping')  # Test connection

                    db = client[db_name]
                    collection = db[collection_name]

                    longitude = float(request.session.get('longitude'))
                    latitude = float(request.session.get('latitude'))
                    start_date = request.session.get('start_date')
                    end_date = request.session.get('end_date')
                    num_contours = analysis_results.get("num_contours", 0)
                    area_percentage = analysis_results.get("area_percentage", 0.0)
                    standard_concentration = pollution_data.get("concentration_standard", 0.0)

                    # IMPORTANT: Use Spanish field names here
                    doc_for_db = {
                        "Fecha de inicio": start_date,
                        "Fecha de recogida": end_date,
                        "Localización longitud": longitude,
                        "Localización latitud": latitude,
                        "Número de contornos detectados": num_contours,
                        "Porcentaje de área detectada": area_percentage,
                        "Concentración estándar": standard_concentration,
                        "Nivel de polución": classification_str,
                        "Imagen de entrada": input_image_b64,
                        "Imagen ROI": roi_image_b64,
                        "Imagen binaria": binary_b64,
                        "Imagen con partículas detectadas": overlay_b64
                    }

                    insert_result = collection.insert_one(doc_for_db)
                    if insert_result.acknowledged:
                        context['success_message'] = "Experimento guardado en la base de datos!"
                    else:
                        context['error_message'] = "Data insertion not acknowledged por MongoDB."

                except Exception as e:
                    context['error_message'] = f"Error saving to MongoDB: {e}"
                    print(f"Error saving to MongoDB: {e}")

                step = 5
            else:
                context['error_message'] = "ROI image or input image not found in session. Please re-upload."
                step = 3

        elif 'reject_roi' in request.POST:
            step = 2

    context['step'] = step

    # For steps where images need to be displayed
    if step == 4:
        context['contour_image_b64'] = request.session.get('contour_image_b64')
        context['roi_image_b64'] = request.session.get('roi_image_b64')

    # For steps where location should be shown
    if step >= 3:
        context['latitude'] = request.session.get('latitude')
        context['longitude'] = request.session.get('longitude')
        context['google_maps_api_key'] = os.environ.get('GOOGLE_MAPS_API_KEY', '')

    # ------------------------------------------------------------------
    # TABLE IN SPANISH SHOWING CONCENTRATION VS. POLLUTION LEVEL
    # ------------------------------------------------------------------
    context['pollution_levels_table'] = """
    <div style="margin-top: 1em;">
      <p><strong>Clasificación de la contaminación según la concentración (μg/m³):</strong></p>
      <table style="border-collapse: collapse; width: 50%; text-align: center;">
        <thead>
          <tr style="background-color: #f2f2f2;">
            <th style="border: 1px solid #ccc; padding: 8px;">Concentración μg/m³</th>
            <th style="border: 1px solid #ccc; padding: 8px;">Nivel de Polución</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style="border: 1px solid #ccc; padding: 8px;">0 - 10</td>
            <td style="border: 1px solid #ccc; padding: 8px;">Muy Bueno</td>
          </tr>
          <tr>
            <td style="border: 1px solid #ccc; padding: 8px;">10 - 20</td>
            <td style="border: 1px solid #ccc; padding: 8px;">Bueno</td>
          </tr>
          <tr>
            <td style="border: 1px solid #ccc; padding: 8px;">20 - 50</td>
            <td style="border: 1px solid #ccc; padding: 8px;">Moderado</td>
          </tr>
          <tr>
            <td style="border: 1px solid #ccc; padding: 8px;">50 - 100</td>
            <td style="border: 1px solid #ccc; padding: 8px;">Malo</td>
          </tr>
          <tr>
            <td style="border: 1px solid #ccc; padding: 8px;">100 - 150</td>
            <td style="border: 1px solid #ccc; padding: 8px;">Muy Malo</td>
          </tr>
          <tr>
            <td style="border: 1px solid #ccc; padding: 8px;">&gt; 150</td>
            <td style="border: 1px solid #ccc; padding: 8px;">Extremo</td>
          </tr>
        </tbody>
      </table>
    </div>
    """

    return render(request, 'particle_analysis.html', context)


def map_contributions(request):
    """
    View to display grouped map markers from MongoDB data.
    """
    context = {}
    try:
        load_environment_variables()

        mongo_username = os.environ.get('MONGO_USERNAME')
        mongo_password = os.environ.get('MONGO_PASSWORD')
        mongo_host = os.environ.get('MONGO_HOST', 'localhost')
        mongo_port = os.environ.get('MONGO_PORT', '27017')
        db_name = os.environ.get('DB_NAME', 'air_quality')
        collection_name = os.environ.get('COLLECTION_NAME', 'results')
        google_maps_api_key = os.environ.get('GOOGLE_MAPS_API_KEY')

        # Check required environment variables
        missing_vars = []
        if not mongo_username:
            missing_vars.append('MONGO_USERNAME')
        if not mongo_password:
            missing_vars.append('MONGO_PASSWORD')
        if not google_maps_api_key:
            missing_vars.append('GOOGLE_MAPS_API_KEY')
        if missing_vars:
            raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

        # Validate port
        if not str(mongo_port).isdigit():
            raise ValueError(f"Invalid port number: {mongo_port}")

        # Build Mongo URI
        mongo_username_encoded = quote_plus(mongo_username)
        mongo_password_encoded = quote_plus(mongo_password)
        mongo_uri = f"mongodb://{mongo_username_encoded}:{mongo_password_encoded}@{mongo_host}:{mongo_port}/"

        # Attempt connection
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("MongoDB connection successful!")

        db = client[db_name]
        collection = db[collection_name]

        # IMPORTANT: Query using Spanish field names you used for insertion
        all_docs = collection.find(
            {},
            {
                'Localización longitud': 1,
                'Localización latitud': 1,
                'Fecha de inicio': 1,
                'Fecha de recogida': 1,
                'Concentración estándar': 1,
                'Nivel de polución': 1,
                '_id': 0
            }
        )

        grouped_data = {}
        for doc in all_docs:
            lat = doc.get('Localización latitud')
            lng = doc.get('Localización longitud')
            if lat is None or lng is None:
                continue

            start_date = doc.get('Fecha de inicio')
            end_date = doc.get('Fecha de recogida')
            std_conc = doc.get('Concentración estándar', 'N/A')
            pol_level = doc.get('Nivel de polución', 'Unknown')

            # Build a dictionary for each sample
            doc_info = {
                'experiment_start_date': start_date,
                'experiment_end_date': end_date,
                'standard_concentration': std_conc,
                'pollution_level': pol_level
            }

            # Use float casting for lat/lng to group properly
            key = (float(lat), float(lng))
            grouped_data.setdefault(key, []).append(doc_info)

        # Prepare final marker data
        marker_data = []
        for (lat, lng), items in grouped_data.items():
            # Sort descending by 'experiment_end_date' for each group
            items_sorted = sorted(
                items,
                key=lambda x: x.get('experiment_end_date', ''),
                reverse=True
            )
            last_sample = items_sorted[0] if items_sorted else None

            marker_info = {
                'latitude': lat,
                'longitude': lng,
                'count': len(items_sorted),
                'all_samples': items_sorted,
                'last_sample': last_sample
            }
            marker_data.append(marker_info)

        # Map center
        if marker_data:
            map_center_latitude = marker_data[0]['latitude']
            map_center_longitude = marker_data[0]['longitude']
        else:
            map_center_latitude = 0
            map_center_longitude = 0

        context.update({
            'google_maps_api_key': google_maps_api_key,
            'map_center_latitude': map_center_latitude,
            'map_center_longitude': map_center_longitude,
            'marker_data': marker_data,
        })

    except Exception as e:
        context['error_message'] = f"Error connecting to MongoDB or reading data: {e}"
        print(f"Error connecting to MongoDB: {e}")

    # Make sure the template name matches your file name
    return render(request, 'map_contributions.html', context)
