<template>
    <va-data-table :items="results" :columns="columns">
    </va-data-table>
</template>

<script setup>
import { ref, defineProps, computed } from 'vue';
const props = defineProps(['results']);

const columns = ref([
    { key: "model" },
    { key: "label" },
    { key: "percentage" }
]);

function formatResults(results) {
    let tmp = [];
    for(const model in results){
        let model_result = {
            model : model.toUpperCase(),
            label: results[model][0][0],
            percentage: results[model][0][1]
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