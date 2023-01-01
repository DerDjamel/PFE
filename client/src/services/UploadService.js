import flaskApp from "./flask";

export default {
    upload(image){
        return flaskApp.post("image", image,  {
            headers: {'Content-Type': 'multipart/form-data'}
        });
    }
}