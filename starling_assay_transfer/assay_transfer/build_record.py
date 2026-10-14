"""Build record datasets through the named scientific stages."""
from .storage import digest,verify_inputs
from .build import arguments,selected_inputs,supported,dependencies,run_stage,write_receipt,checked_stage
from .canonical import task_contract
from .filtering import prepare_record_pool
from .sampling import sample_record
from .augmentation import augment_record
from .targets import generate
from .dataset import create_dataset,upload_verified
from .verify import verify_task


def execute(args,inputs,task):
    policy=task_contract(task,args.contract,split_disjointness=args.split_disjointness)
    supported(policy)
    files=inputs['tasks'][task]['files']
    root=args.output_root/'record'/policy['slug']
    base=dependencies(inputs,policy,args.smoke,'record')
    commands=('pool','sample','augment','targets','dataset','verify') if args.command=='all' else (args.command,)
    for command in commands:
        if command=='pool':
            run_stage(root,'pool',base,[],lambda out:prepare_record_pool(out,files,policy,args.smoke),args.resume)
        elif command=='sample':
            run_stage(root,'sampling',base,['pool'],lambda out:sample_record(out,root/'pool',policy,args.smoke),args.resume)
        elif command=='augment':
            run_stage(root,'augmentation',base,['sampling'],
                      lambda out:augment_record(out,root/'pool',root/'sampling',policy,args.smoke),args.resume)
        elif command=='targets':
            run_stage(root,'targets',base,['augmentation'],
                      lambda out:generate(out,root,policy,'record',args.smoke),args.resume)
        elif command=='dataset':
            run_stage(root,'dataset/record',base,['targets'],
                      lambda out:create_dataset(out,root,policy,'record',args.render_workers),args.resume)
        elif command=='verify':
            checked_stage(root/'dataset/record',base,root)
            receipt=verify_task(root,policy,'record',args.render_workers)
            run_stage(root,'verification/record',base,['dataset/record'],
                      lambda out:write_receipt(out,receipt),args.resume)
        elif command=='upload':
            checked_stage(root/'verification/record',base,root)
            uploaded=upload_verified(root,policy,'record',args.repo_id,args.revision,args.private)
            print(uploaded['commit_url'],flush=True)
    print(f'{task}: record {args.command} finished ({base["scope"]})',flush=True)


def main():
    args=arguments('record')
    inputs=selected_inputs(args)
    for task in inputs['tasks']:execute(args,inputs,task)
    verify_inputs(inputs)


if __name__=='__main__':main()
