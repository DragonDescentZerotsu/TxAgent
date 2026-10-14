"""Build context datasets through the named scientific stages."""
from .storage import digest,verify_inputs
from .build import arguments,selected_inputs,supported,dependencies,run_stage,write_receipt,checked_stage
from .canonical import task_contract
from .filtering import prepare_context_pool
from .sampling import sample_context
from .augmentation import augment_context
from .targets import generate
from .dataset import create_dataset,upload_verified
from .verify import verify_task


def execute(args,inputs,task):
    policy=task_contract(task,args.contract,split_disjointness=args.split_disjointness,augmentation_ratio=args.augmentation_ratio if args.augment else 0.0)
    supported(policy)
    files=inputs['tasks'][task]['files']
    root=args.output_root/'context_molecule'/policy['slug']
    base=dependencies(inputs,policy,args.smoke,'context')
    commands=('pool','sample','augment','targets','dataset','verify') if args.command=='all' else (args.command,)
    for command in commands:
        if command=='pool':
            run_stage(root,'pool',base,[],lambda out:prepare_context_pool(out,files,policy,args.smoke),args.resume)
        elif command=='sample':
            run_stage(root,'sampling',base,['pool'],lambda out:sample_context(out,root/'pool',policy,args.smoke),args.resume)
        elif command=='augment':
            run_stage(root,'augmentation',base,['sampling'],
                      lambda out:augment_context(out,root/'sampling',policy,args.smoke),args.resume)
        elif command=='targets':
            run_stage(root,'targets',base,['augmentation'],
                      lambda out:generate(out,root,policy,'context',args.smoke),args.resume)
        elif command=='dataset':
            run_stage(root,'dataset/context',base,['targets'],
                      lambda out:create_dataset(out,root,policy,'context',args.render_workers),args.resume)
        elif command=='verify':
            checked_stage(root/'dataset/context',base,root)
            receipt=verify_task(root,policy,'context',args.render_workers)
            run_stage(root,'verification/context',base,['dataset/context'],
                      lambda out:write_receipt(out,receipt),args.resume)
        elif command=='upload':
            checked_stage(root/'verification/context',base,root)
            uploaded=upload_verified(root,policy,'context',args.repo_id,args.revision,args.private)
            print(uploaded['commit_url'],flush=True)
    print(f'{task}: context {args.command} finished ({base["scope"]})',flush=True)


def main():
    args=arguments('context')
    inputs=selected_inputs(args,gold_only=not args.augment or args.augmentation_ratio==0)
    for task in inputs['tasks']:execute(args,inputs,task)
    verify_inputs(inputs)


if __name__=='__main__':main()
