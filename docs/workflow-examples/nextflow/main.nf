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
          --require-converged \
          --require-final-check-ok
        """
}

process inspect {
    tag 'gpurec-inspect'
    input:
        path summary from optimize.out.summary
        path checkpoint from optimize.out.best_ckpt
    output:
        path('inspect.ok'), emit: ready
    script:
        """
        gpurec summary-info --summary ${summary} \
          --require-converged \
          --require-final-check-ok
        gpurec checkpoint-info --checkpoint ${checkpoint} \
          --require-final-check-ok
        touch inspect.ok
        """
}

process sample {
    tag 'gpurec-sample'
    input:
        path ready from inspect.out.ready
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
    inspected = inspect(optimized.out.summary, optimized.out.best_ckpt)
    sample(optimized.out.best_ckpt)
}
