#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.run_config = file('../end-to-end-tutorial/run.json')
params.sample_count = 2
params.sample_seed = 42

process validateConfig {
    tag 'gpurec-validate'
    output:
        path('validate.ok'), emit: ready
    script:
        """
        gpurec validate-config \
          --config ${params.run_config} \
          --check-preprocess \
          --require-mode-default-optimizer \
          --require-cuda-backward-ready
        touch validate.ok
        """
}

process optimize {
    tag 'gpurec-optimize'
    input:
        path ready from validateConfig.out.ready
    output:
        path('output_gpurec/checkpoints/best.pt'), emit: best_ckpt
        path('output_gpurec/summary.json'), emit: summary
    script:
        """
        gpurec optimize \
          --config ${params.run_config} \
          --require-mode-default-optimizer \
          --require-final-check-ok
        """
}

process sample {
    tag 'gpurec-sample'
    input:
        path checkpoint from optimize.out.best_ckpt
    output:
        path 'output_gpurec/reconciliations'
    script:
        """
        gpurec sample \
          --checkpoint ${checkpoint} \
          --samples ${params.sample_count} \
          --seed ${params.sample_seed} \
          --sample-out-dir output_gpurec
        """
}

workflow {
    validate = validateConfig()
    optimized = optimize(validate.out.ready)
    sample(optimized.out.best_ckpt)
}
