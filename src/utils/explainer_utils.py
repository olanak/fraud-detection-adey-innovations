import shap
from pathlib import Path

def save_shap_plots(model, X_test, output_dir):
    """Save SHAP plots to directory"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Summary plot
    shap.summary_plot(shap_values[1], X_test, show=False)
    plt.savefig(Path(output_dir)/'summary_plot.png')
    plt.close()
    
    return {
        'shap_values': shap_values,
        'expected_value': explainer.expected_value[1]
    }

def generate_lime_explanation(model, instance, feature_names, class_names, output_path):
    """Generate and save LIME explanation"""
    explainer = lime_tabular.LimeTabularExplainer(
        instance,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    
    exp = explainer.explain_instance(
        instance,
        model.predict_proba,
        num_features=10
    )
    
    exp.save_to_file(output_path)