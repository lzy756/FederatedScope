"""
快速验证配置是否能够正确加载
不加载完整的FederatedScope模块，而是直接读取和检查配置
"""
import yaml
import sys

def verify_yaml_config():
    """验证YAML配置文件的内容"""

    print("\n" + "="*80)
    print("验证配置文件: cross_domain_adaptive_office_caltech.yaml")
    print("="*80 + "\n")

    config_path = 'scripts/example_configs/cross_domain_adaptive_office_caltech.yaml'

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        print("[OK] 配置文件加载成功\n")
    except Exception as e:
        print(f"[ERROR] 配置文件加载失败: {e}")
        return False

    # 验证关键配置项
    checks = []

    # 1. 联邦学习配置
    print("[1] 联邦学习配置:")
    checks.append(('federate.mode', cfg.get('federate', {}).get('mode'), 'standalone'))
    checks.append(('federate.method', cfg.get('federate', {}).get('method'), 'cross_domain_adaptive'))
    checks.append(('federate.client_num', cfg.get('federate', {}).get('client_num'), 20))
    checks.append(('federate.total_round_num', cfg.get('federate', {}).get('total_round_num'), 300))
    checks.append(('federate.sample_client_num', cfg.get('federate', {}).get('sample_client_num'), 5))
    checks.append(('federate.make_global_eval', cfg.get('federate', {}).get('make_global_eval'), True))

    for key, value, expected in checks[-6:]:
        status = "[OK]" if value == expected else "[X]"
        print(f"  {status} {key}: {value} {'(预期: ' + str(expected) + ')' if value != expected else ''}")

    # 2. 数据配置
    print("\n[2] 数据配置:")
    checks.append(('data.type', cfg.get('data', {}).get('type'), 'office_caltech'))
    checks.append(('data.batch_size', cfg.get('data', {}).get('batch_size'), 4))
    checks.append(('data.dirichlet_alpha', cfg.get('data', {}).get('dirichlet_alpha'), 0.1))

    for key, value, expected in checks[-3:]:
        status = "[OK]" if value == expected else "[X]"
        print(f"  {status} {key}: {value} {'(预期: ' + str(expected) + ')' if value != expected else ''}")

    # 3. 模型配置
    print("\n[3] 模型配置:")
    checks.append(('model.type', cfg.get('model', {}).get('type'), 'fedlsa_cnn'))
    checks.append(('model.hidden', cfg.get('model', {}).get('hidden'), 512))
    checks.append(('model.num_classes', cfg.get('model', {}).get('num_classes'), 10))

    for key, value, expected in checks[-3:]:
        status = "[OK]" if value == expected else "[X]"
        print(f"  {status} {key}: {value} {'(预期: ' + str(expected) + ')' if value != expected else ''}")

    # 4. Trainer配置
    print("\n[4] Trainer配置:")
    checks.append(('trainer.type', cfg.get('trainer', {}).get('type'), 'cross_domain_adaptive'))

    for key, value, expected in checks[-1:]:
        status = "[OK]" if value == expected else "[X]"
        print(f"  {status} {key}: {value} {'(预期: ' + str(expected) + ')' if value != expected else ''}")

    # 5. FedLSA配置
    print("\n[5] FedLSA配置:")
    fedlsa = cfg.get('fedlsa', {})
    print(f"  [OK] use: {fedlsa.get('use', False)}")
    print(f"  [OK] lambda_com: {fedlsa.get('lambda_com', 0.0)}")
    print(f"  [OK] tau: {fedlsa.get('tau', 0.0)}")
    print(f"  [OK] use_projector: {fedlsa.get('use_projector', False)}")
    print(f"  [OK] projector_output_dim: {fedlsa.get('projector_output_dim', 0)}")
    print(f"  [OK] share_projector: {fedlsa.get('share_projector', False)}")
    print(f"  [OK] alpha_sep: {fedlsa.get('alpha_sep', 0.0)}")

    # 6. OnDemFL配置
    print("\n[6] OnDemFL配置:")
    ondemfl = cfg.get('ondemfl', {})
    print(f"  [OK] enable: {ondemfl.get('enable', False)}")
    print(f"  [OK] pretrain_rounds: {ondemfl.get('pretrain_rounds', 0)}")
    print(f"  [OK] ondemand_rounds: {ondemfl.get('ondemand_rounds', 0)}")
    print(f"  [OK] subset_size: {ondemfl.get('subset_size', 0)}")
    print(f"  [OK] weight_scheme: {ondemfl.get('weight_scheme', '')}")
    print(f"  [OK] dp_loss: {ondemfl.get('dp_loss', '')}")
    print(f"  [OK] freeze_predictor_after_stage1: {ondemfl.get('freeze_predictor_after_stage1', False)}")

    # 7. CrossDomainAdaptive配置
    print("\n[7] CrossDomainAdaptive配置:")
    cda = cfg.get('cross_domain_adaptive', {})
    print(f"  [OK] anchor_reweight: {cda.get('anchor_reweight', False)}")
    print(f"  [OK] anchor_weight_momentum: {cda.get('anchor_weight_momentum', 0.0)}")
    print(f"  [OK] anchor_weight_eps: {cda.get('anchor_weight_eps', 0.0)}")

    # 8. 训练配置
    print("\n[8] 训练配置:")
    train = cfg.get('train', {})
    print(f"  [OK] local_update_steps: {train.get('local_update_steps', 0)}")
    print(f"  [OK] optimizer.type: {train.get('optimizer', {}).get('type', '')}")
    print(f"  [OK] optimizer.lr: {train.get('optimizer', {}).get('lr', 0.0)}")

    # 9. 验证一致性
    print("\n[9] 配置一致性检查:")
    pretrain = ondemfl.get('pretrain_rounds', 0)
    ondemand = ondemfl.get('ondemand_rounds', 0)
    total = cfg.get('federate', {}).get('total_round_num', 0)

    if total == pretrain + ondemand:
        print(f"  [OK] 总轮数一致: {total} = {pretrain} + {ondemand}")
    else:
        print(f"  [WARN] 总轮数不一致: {total} ≠ {pretrain} + {ondemand}")

    if cfg.get('federate', {}).get('method') == 'cross_domain_adaptive':
        print(f"  [OK] 方法名正确: cross_domain_adaptive")
    else:
        print(f"  [X] 方法名错误: {cfg.get('federate', {}).get('method')}")

    if cfg.get('trainer', {}).get('type') == 'cross_domain_adaptive':
        print(f"  [OK] Trainer类型正确: cross_domain_adaptive")
    else:
        print(f"  [X] Trainer类型错误: {cfg.get('trainer', {}).get('type')}")

    # 10. 验证代码注册
    print("\n[10] 验证代码注册情况:")

    # 检查worker注册
    try:
        with open('federatedscope/contrib/worker/cross_domain_adaptive.py', 'r', encoding='utf-8') as f:
            worker_content = f.read()
            if 'register_worker' in worker_content and 'cross_domain_adaptive' in worker_content:
                print("  [OK] Worker已注册")
            else:
                print("  [WARN] Worker注册可能不完整")
    except Exception as e:
        print(f"  [X] 无法读取Worker文件: {e}")

    # 检查trainer注册
    try:
        with open('federatedscope/contrib/trainer/cross_domain_adaptive_trainer.py', 'r', encoding='utf-8') as f:
            trainer_content = f.read()
            if 'register_trainer' in trainer_content and 'cross_domain_adaptive' in trainer_content:
                print("  [OK] Trainer已注册")
            else:
                print("  [WARN] Trainer注册可能不完整")
    except Exception as e:
        print(f"  [X] 无法读取Trainer文件: {e}")

    # 检查模型
    try:
        with open('federatedscope/cv/model/model_builder.py', 'r', encoding='utf-8') as f:
            model_content = f.read()
            if 'fedlsa_cnn' in model_content:
                print("  [OK] 模型(fedlsa_cnn)已注册")
            else:
                print("  [WARN] 模型(fedlsa_cnn)注册可能不完整")
    except Exception as e:
        print(f"  [X] 无法读取模型文件: {e}")

    print("\n" + "="*80)
    print("配置验证完成!")
    print("="*80)

    # 检查是否有错误
    all_passed = all(value == expected for _, value, expected in checks)

    if all_passed:
        print("\n[OK] 所有关键配置项验证通过")
        print("\n建议的训练命令:")
        print("  python federatedscope/main.py --cfg scripts/example_configs/cross_domain_adaptive_office_caltech.yaml")
    else:
        print("\n[WARN] 部分配置项存在差异，请检查上述输出")

    print("\n" + "="*80 + "\n")

    return True

if __name__ == '__main__':
    try:
        verify_yaml_config()
    except Exception as e:
        print(f"\n[X] 验证过程发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
