// components/geolocation.js

// Escucha el evento 'message' de Streamlit
window.addEventListener("message", event => {
    // Solo procesa mensajes que son de Streamlit
    if (event.source !== window.parent) {
        return;
    }

    const { type, data } = event.data;

    // Cuando Streamlit nos pida la ubicación
    if (type === "streamlit:requestLocation") {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    const lat = position.coords.latitude;
                    const lon = position.coords.longitude;
                    // Envía la ubicación de vuelta a Streamlit
                    Streamlit.setComponentValue({ latitude: lat, longitude: lon });
                },
                (error) => {
                    let errorMessage;
                    switch(error.code) {
                        case error.PERMISSION_DENIED:
                            errorMessage = "El usuario denegó la solicitud de geolocalización.";
                            break;
                        case error.POSITION_UNAVAILABLE:
                            errorMessage = "Información de ubicación no disponible.";
                            break;
                        case error.TIMEOUT:
                            errorMessage = "La solicitud para obtener la ubicación del usuario ha caducado.";
                            break;
                        case error.UNKNOWN_ERROR:
                            errorMessage = "Un error desconocido ocurrió.";
                            break;
                    }
                    // Envía el error de vuelta a Streamlit
                    Streamlit.setComponentValue({ error: errorMessage });
                },
                {
                    enableHighAccuracy: true, // Solicita la mejor precisión posible
                    timeout: 10000,           // Tiempo máximo en ms para obtener la ubicación
                    maximumAge: 0             // No usar una posición almacenada en caché
                }
            );
        } else {
            // El navegador no soporta geolocalización
            Streamlit.setComponentValue({ error: "Geolocalización no soportada por este navegador." });
        }
    }
});

// Le dice a Streamlit que el componente está listo
Streamlit.setComponentReady();