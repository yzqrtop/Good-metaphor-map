"""
Ablation Study Example

Demonstrates how to run GoodM ablation experiments
"""

from goodm import GoodMTuple, MEF, AblationStudy


def create_sample_data():
    """Create sample test data"""
    samples = [
        GoodMTuple(
            E="竹篮",
            V="一场空",
            EE="用竹子编织的篮子，质地疏松，无法盛水",
            VE="什么都没有，完全落空的状态",
            C="无法保持或留住",
            Sc="努力付出但没有任何收获的场景",
            Sy="徒劳无功，白费力气"
        ),
        GoodMTuple(
            E="泥菩萨",
            V="过江自身难保",
            EE="用泥土塑造的菩萨像，遇水会融化",
            VE="连自己都保护不了的状态",
            C="脆弱，无法自保",
            Sc="自身处于危险境地无法帮助他人",
            Sy="自顾不暇，无力帮助他人"
        ),
        GoodMTuple(
            E="黄鼠狼",
            V="给鸡拜年没安好心",
            EE="一种食肉动物，鸡的天敌",
            VE="假装友好但心怀恶意",
            C="表面友好，实际有害",
            Sc="伪装善意但实际图谋不轨的场景",
            Sy="警惕伪装的恶意"
        )
    ]
    return samples


def main():
    print("=" * 60)
    print("GoodM Ablation Study Example")
    print("=" * 60)
    
    # Create test data
    print("\n[1] Create test data")
    print("-" * 40)
    test_tuples = create_sample_data()
    print(f"Created {len(test_tuples)} test samples")
    for i, t in enumerate(test_tuples, 1):
        print(f"  Sample {i}: {t.E} - {t.V}")
    
    # Initialize evaluator and ablation study
    print("\n[2] Initialize MEF evaluator and ablation study")
    print("-" * 40)
    mef = MEF()
    ablation = AblationStudy(mef)
    
    # Run leave-one-out ablation experiment
    print("\n[3] Leave-one-out ablation experiment")
    print("-" * 40)
    loo_results = ablation.leave_one_out(test_tuples)
    
    print(f"Baseline performance: {loo_results['baseline']:.4f}")
    print("\nAblation results for each component:")
    for component, result in loo_results['ablations'].items():
        print(f"  Remove {component}:")
        print(f"    Performance: {result['score_without']:.4f}")
        print(f"    Drop: {result['performance_drop']:.4f} ({result['drop_percentage']:.2f}%)")
    
    # Run key components removal experiment
    print("\n[4] Remove key components experiment")
    print("-" * 40)
    key_results = ablation.remove_key_components(test_tuples)
    
    print(f"Baseline performance: {key_results['baseline']:.4f}")
    print(f"Performance after removing {key_results['removed_components']}: {key_results['without_key_components']:.4f}")
    print(f"Performance drop: {key_results['performance_drop']:.4f} ({key_results['drop_percentage']:.2f}%)")
    
    # Run pairwise combination experiment (simplified display)
    print("\n[5] Pairwise combination experiment")
    print("-" * 40)
    pairwise_results = ablation.pairwise_combination(test_tuples)
    
    print(f"Total tested combinations: {len(pairwise_results['combinations'])}")
    print(f"Best combination: {' + '.join(pairwise_results['best_combination'])}")
    print(f"Best score: {pairwise_results['best_score']:.4f}")
    
    # Display top 5 combinations
    sorted_combos = sorted(
        pairwise_results['combinations'].items(),
        key=lambda x: x[1]['score'],
        reverse=True
    )[:5]
    
    print("\nTop 5 combinations:")
    for combo_name, combo_data in sorted_combos:
        print(f"  {combo_name}: {combo_data['score']:.4f}")
    
    print("\n" + "=" * 60)
    print("Ablation study completed")
    print("=" * 60)


if __name__ == "__main__":
    main()