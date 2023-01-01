<template>
    <va-data-table :items="results" :columns="columns">
    </va-data-table>
</template>

<script setup>
import { ref, defineProps, computed } from 'vue';
const props = defineProps(['results']);

const columns = ref([
    { key: "model" },
    { key: "only" },
    { key: "pca" },
    { key: "umap" },
]);

function formatResults(results) {
    let tmp = [];
    for(const model in results){
        let model_result = {
            model : model.toUpperCase(),
            only: results[model].Only   || "NA",
            pca: results[model].PCA     || "NA",
            umap: results[model].UMAP   || "NA",
        };
        tmp.push(model_result);
    }
    return tmp
}

const results = computed(() => {
    return formatResults(props.results);
})


</script>

<style lang="scss" scoped>
</style>