<template>
    <div class="row justify--center mt-4">
        <div class="flex md6">
            <va-card square outlined stripe stripe-color="primary">
                <va-card-title class="display-6">Make a Prediction Form.</va-card-title>
                <va-divider />
                <div class="row px-3 justify--center">
                    <va-card square outlined :bordered="false" class="flex">
                        <div class="row pa-3 justify--center">
                            <va-form style="width: 500px;">
                                <va-file-upload upload-button-text="Upload X-ray image" class="mb-2" v-model="image"
                                    type="gallery" file-types="image/*" />
                                <va-divider />
                                <div class="title my-2">Patient's Full Name</div>
                                <va-input outline style="width: 500px;" class="mb-4" v-model="patient_name" />

                                <div class="title my-2">Methods</div>
                                <va-select bordered outline style="width: 500px;" class="mb-4" v-model="method"
                                    :options="methods" />
                                <div class="title my-2">Algorithms</div>
                                <va-select bordered outline :disabled="!clustering_selected" style="width: 500px;"
                                    class="mb-4" v-model="algorithm" :options="algorithms" />

                                <div class="title my-2">Dimensionality reduction</div>
                                <va-checkbox :disabled="!clustering_selected" class="mb-2"
                                    v-model="dim_reduction_methods" array-value="Only" label="No PCA/UMAP" />
                                <va-checkbox :disabled="!clustering_selected" class="mb-2"
                                    v-model="dim_reduction_methods" array-value="PCA" label="PCA" />
                                <va-checkbox :disabled="!clustering_selected" class="mb-2"
                                    v-model="dim_reduction_methods" array-value="UMAP" label="UMAP" />

                                <div class="title mb-2 mt-4">Models</div>
                                <va-checkbox class="mb-2" v-model="models" array-value="vgg16" label="VGG-16" />
                                <va-checkbox class="mb-2" v-model="models" array-value="resnet50" label="ResNet-50" />
                                <va-checkbox class="mb-2" v-model="models" array-value="inceptionv3"
                                    label="Inception V3" />
                                <va-checkbox class="mb-2" v-model="models" array-value="xception" label="Xception" />

                            </va-form>
                        </div>
                    </va-card>
                </div>
                <va-divider />
                <va-card-actions align="center">
                    <va-button :rounded="false" @click="$emit('sendPredictionData', payload)">Submit</va-button>
                </va-card-actions>
            </va-card>
        </div>
    </div>
</template>

<script setup>
import { ref, computed } from "vue";
// data
const image = ref([]);
const models = ref([]);
const method = ref("Classification");
const algorithm = ref("K-MEANS");
const patient_name = ref("");
const dim_reduction_methods = ref([]);


const methods = ref(["Classification", "Clustering"]);
const algorithms = ref(["K-MEANS", "F-CMEANS"]);

// computed 
const clustering_selected = computed(() => {
    return method.value === "Clustering"
})

const payload = {
    method,
    algorithm,
    patient_name,
    models,
    dim_reduction_methods,
    image
}

</script>

<style>
</style>