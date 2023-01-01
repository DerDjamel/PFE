<template>
    <PredictionForm @sendPredictionData="predict"></PredictionForm>
    <va-modal size="large" blur v-model="showModal" :message="modelMessage" title="Prediction Progress"
        hide-default-actions>
        <va-progress-bar v-if="!finished" indeterminate />
        <template #footer>
            <va-button @click="showModal = false" :disabled="modelButton">
                Done
            </va-button>
        </template>
    </va-modal>
    <section class="ma-4" v-if="finished && results">
        <div class="row justify--center mt-4">
            <va-card square outlined stripe stripe-color="primary">
                <div class="pa-4 results-container">
                    <h3 class="display-3 py-2">Results</h3>
                    <va-divider />
                    <ClassificationTable v-if="finished && results && method === 'Classification'" :results="results"></ClassificationTable>
                    <ClusteringTable v-if="finished && results && method === 'Clustering'" :results="results"></ClusteringTable>
                </div>
            </va-card>
        </div>
    </section>
</template>

<script setup>
import PredictionForm from '../components/PredictionForm.vue';
import UploadService from '../services/UploadService';
import PredictionService from "../services/PredictionService";
import ClassificationTable from '../components/ClassificationTable.vue';
import ClusteringTable from '../components/ClusteringTable.vue';

import { ref } from 'vue'

const results = ref({});
const showModal = ref(false);
const modelMessage = ref("");
const modelButton = ref(true);
const finished = ref(false);
const method = ref('');

async function predict(data) {
    let uploadedImage;
    // start the loading in the modal 
    showModal.value = true;
    modelMessage.value = "Uploading image...";
    finished.value = false;
    method.value = data.method.value;
    // upload image
    let image = new FormData();
    image.append('image', data.image.value[0]);
    image.append('patient_name', data.patient_name.value)
    try {
        const response = await UploadService.upload(image);
        uploadedImage = response.data;
        modelMessage.value = "Image has been Uploaded.";
    } catch (error) {
        console.error(`FAILURE!! ${error}`);
        modelMessage.value = "Error : Image has could not be uploaded.";
        finished.value = true;
        modelButton.value = false;
        return;
    }
    
    modelMessage.value = "Models are predicting ..."
    // send the prediction data to the corresponding route
    if (method.value === "Classification") {
        try {
            const response = await PredictionService.classification({
                "image": uploadedImage.image,
                "patient_name": uploadedImage.patient_name,
                "models": data.models.value
            })
            results.value = response.data
            modelMessage.value = "Models have finished."

        } catch (error) {
            console.log(error);
            modelMessage.value = "Something went wrong."
        }
    } else {
        try {
            const response = await PredictionService.clustering({
                "image": uploadedImage.image,
                "patient_name": uploadedImage.patient_name,
                "models": data.models.value,
                "algorithm": data.algorithm.value,
                "dim_reduction_methods": data.dim_reduction_methods.value
            })
            results.value = response.data
            modelMessage.value = "Models have finished."
        } catch (error) {
            console.log(error);
            modelMessage.value = "Something went wrong."
        }
    }
    modelButton.value = false;
    finished.value = true;

}




</script>

<style >
.results-container {
    width: 673px
}
</style>