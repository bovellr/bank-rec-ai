"""
Bank Reconciliation AI - Main Application 
Intelligent bank reconciliation with machine learning.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Fix import paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'app'))

# Now import your modules
try:
    from self_learning import SelfLearningManager
    from reconcile import run_reconciliation_with_learning
    from trainer import train_model
    import config
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please check that all required files exist in the correct locations")
    st.stop()



# Add this function to create the self-learning interface
def create_self_learning_section():
    """
    Create the self-learning interface in Streamlit.
    """
    st.header("🧠 Self-Learning Features")
    
    # Initialize learning manager
    if 'learning_manager' not in st.session_state:
        st.session_state.learning_manager = SelfLearningManager()
    
    learning_manager = st.session_state.learning_manager
    
    # Get learning statistics
    stats = learning_manager.get_learning_statistics()
    
    # Display current learning status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Feedback Entries",
            value=stats['total_feedback_entries'],
            help="Total number of human feedback entries collected"
        )
    
    with col2:
        st.metric(
            label="Model Retrains",
            value=stats['total_retrains'],
            help="Number of times the model has been retrained"
        )
    
    with col3:
        st.metric(
            label="Current Accuracy",
            value=f"{stats['latest_accuracy']:.1%}",
            delta=stats['accuracy_trend'],
            help="Latest model accuracy on feedback data"
        )
    
    with col4:
        st.metric(
            label="Pending Reviews",
            value=stats['pending_uncertain_cases'],
            help="Cases waiting for human review"
        )
    
    # Create tabs for different self-learning features
    tab1, tab2, tab3 = st.tabs(["📝 Review Cases", "📊 Performance", "🔄 Retrain Model"])
    
    with tab1:
        st.subheader("Cases Needing Human Review")
        
        # Load uncertain cases
        uncertain_cases = learning_manager.load_uncertain_cases()
        
        if not uncertain_cases:
            st.info("🎉 No uncertain cases found! The model is confident in all its predictions.")
        else:
            st.write(f"Found **{len(uncertain_cases)}** cases that need human review:")
            
            # Display cases for review
            for i, case in enumerate(uncertain_cases[:10]):  # Show top 10 most uncertain
                with st.expander(
                    f"Case {i+1}: {case['reason']} (Confidence: {case['confidence']:.2f})", 
                    expanded=(i == 0)  # Expand first case by default
                ):
                    
                    # Show case details
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Model Prediction:**")
                        prediction_text = "✅ Match" if case['prediction'] == 1 else "❌ No Match"
                        st.write(prediction_text)
                        st.write(f"**Confidence:** {case['confidence']:.3f}")
                        st.write(f"**Uncertainty:** {case['uncertainty']:.3f}")
                        st.write(f"**Reason:** {case['reason']}")
                    
                    with col2:
                        st.write("**Your Decision:**")
                        
                        # User feedback buttons
                        feedback_col1, feedback_col2 = st.columns(2)
                        
                        with feedback_col1:
                            if st.button(
                                "✅ Confirm Match", 
                                key=f"confirm_{case['index']}",
                                help="These transactions are indeed a match"
                            ):
                                # Collect feedback
                                success = collect_case_feedback(case, 1, learning_manager)
                                if success:
                                    st.success("✅ Feedback recorded! This will help improve the model.")
                                    st.rerun()
                        
                        with feedback_col2:
                            if st.button(
                                "❌ Reject Match", 
                                key=f"reject_{case['index']}",
                                help="These transactions are NOT a match"
                            ):
                                # Collect feedback
                                success = collect_case_feedback(case, 0, learning_manager)
                                if success:
                                    st.success("✅ Feedback recorded! This will help improve the model.")
                                    st.rerun()
                    
                    # Optional comment
                    user_comment = st.text_input(
                        "Optional comment:", 
                        key=f"comment_{case['index']}",
                        placeholder="Why did you make this decision?"
                    )
                    
                    # Show transaction details if available
                    if 'bank_index' in case and 'erp_index' in case:
                        if st.checkbox(f"Show transaction details", key=f"details_{case['index']}"):
                            try:
                                # This would need access to the original data
                                # You'll need to modify this based on how you store the data
                                st.write("**Transaction Details:**")
                                st.write("(Details would be shown here - need to pass transaction data)")
                            except:
                                st.write("Transaction details not available")
            
            # Bulk feedback options
            st.subheader("Bulk Actions")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Refresh Uncertain Cases"):
                    st.rerun()
            
            with col2:
                if st.button("✨ Clear All Reviewed Cases"):
                    # This would remove cases that have been reviewed
                    st.success("All reviewed cases cleared!")
    
    with tab2:
        st.subheader("Learning Performance")
        
        # Performance history chart
        if stats['total_retrains'] > 0:
            perf_history = learning_manager.performance_history
            
            if perf_history:
                # Create performance DataFrame
                perf_df = pd.DataFrame(perf_history)
                perf_df['timestamp'] = pd.to_datetime(perf_df['timestamp'])
                
                # Accuracy over time chart
                st.write("**Model Accuracy Over Time:**")
                chart_data = perf_df.set_index('timestamp')[['old_accuracy', 'new_accuracy']]
                st.line_chart(chart_data)
                
                # Improvement metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Recent Improvements:**")
                    for entry in perf_history[-3:]:  # Show last 3 retrains
                        improvement = entry.get('improvement', 0)
                        timestamp = entry.get('timestamp', '')
                        st.write(f"• {timestamp[:10]}: {improvement:+.1%}")
                
                with col2:
                    st.write("**Training Data Growth:**")
                    for entry in perf_history[-3:]:
                        feedback_count = entry.get('feedback_samples', 0)
                        total_count = entry.get('total_samples', 0)
                        st.write(f"• Total samples: {total_count} (+{feedback_count} feedback)")
        else:
            st.info("No retraining history yet. Collect some feedback and retrain the model to see performance metrics!")
        
        # Feedback history
        if stats['total_feedback_entries'] > 0:
            st.subheader("Feedback Summary")
            
            feedback_history = learning_manager.feedback_history
            
            # Create feedback summary
            feedback_df = pd.DataFrame(feedback_history)
            
            if not feedback_df.empty:
                # Feedback over time
                feedback_df['timestamp'] = pd.to_datetime(feedback_df['timestamp'])
                feedback_df['date'] = feedback_df['timestamp'].dt.date
                
                daily_feedback = feedback_df.groupby('date').size()
                st.write("**Daily Feedback Collection:**")
                st.bar_chart(daily_feedback)
                
                # Feedback decisions
                decision_counts = feedback_df['user_decision'].value_counts()
                st.write("**Feedback Decisions:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Confirmed Matches", decision_counts.get(1, 0))
                with col2:
                    st.metric("Rejected Matches", decision_counts.get(0, 0))
    
    with tab3:
        st.subheader("Model Retraining")
        
        # Check if retraining is possible
        feedback_count = stats['total_feedback_entries']
        min_feedback_required = 5
        
        if feedback_count < min_feedback_required:
            st.warning(f"⚠️ Need at least {min_feedback_required} feedback entries to retrain. You have {feedback_count}.")
            st.info("💡 Review some uncertain cases first to collect feedback.")
        else:
            st.success(f"✅ Ready to retrain! You have {feedback_count} feedback entries.")
            
            # Retraining options
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Retraining Settings:**")
                auto_retrain = st.checkbox(
                    "Auto-retrain after 10 feedback entries",
                    value=False,
                    help="Automatically retrain the model when enough feedback is collected"
                )
                
                backup_model = st.checkbox(
                    "Backup current model",
                    value=True,
                    help="Create a backup of the current model before retraining"
                )
            
            with col2:
                st.write("**Expected Impact:**")
                st.info(f"📊 Training samples: {feedback_count} new + existing data")
                st.info(f"🎯 Expected improvement: Variable (depends on feedback quality)")
                
                if stats['latest_accuracy'] > 0:
                    st.info(f"📈 Current accuracy: {stats['latest_accuracy']:.1%}")
            
            # Retrain button
            if st.button("🚀 Retrain Model Now", type="primary"):
                retrain_model_with_feedback(learning_manager)
        
        # Retraining history
        if stats['total_retrains'] > 0:
            st.subheader("Retraining History")
            
            history_df = pd.DataFrame(learning_manager.performance_history)
            
            # Display recent retraining events
            for i, entry in enumerate(learning_manager.performance_history[-5:]):  # Last 5 retrains
                with st.expander(f"Retrain #{i+1} - {entry.get('timestamp', '')[:10]}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Old Accuracy", f"{entry.get('old_accuracy', 0):.1%}")
                    with col2:
                        st.metric("New Accuracy", f"{entry.get('new_accuracy', 0):.1%}")
                    with col3:
                        improvement = entry.get('improvement', 0)
                        st.metric("Improvement", f"{improvement:+.1%}")
                    
                    st.write(f"**Feedback samples:** {entry.get('feedback_samples', 0)}")
                    st.write(f"**Total samples:** {entry.get('total_samples', 0)}")

def collect_case_feedback(case, user_decision, learning_manager):
    """
    Collect feedback for a specific case.
    """
    try:
        # You'll need to have access to the original transaction data
        # This is a simplified version - you may need to modify based on your data structure
        
        # For now, create dummy bank and ERP row data
        # In practice, you'd retrieve this from your session state or database
        bank_row = {
            'Amount': 100.0,  # You'd get this from the actual data
            'Date': '2024-01-01',
            'Description': 'Sample Bank Transaction'
        }
        
        erp_row = {
            'Amount': 100.0,
            'Date': '2024-01-01', 
            'Description': 'Sample ERP Transaction'
        }
        
        # Collect the feedback
        success = learning_manager.collect_feedback(
            case_index=case['index'],
            bank_row=bank_row,
            erp_row=erp_row,
            user_decision=user_decision,
            confidence=case['confidence'],
            user_comment=""
        )
        
        return success
        
    except Exception as e:
        st.error(f"Error collecting feedback: {e}")
        return False

def retrain_model_with_feedback(learning_manager):
    """
    Retrain the model with collected feedback.
    """
    with st.spinner("🔄 Retraining model with feedback..."):
        try:
            success = learning_manager.retrain_model()
            
            if success:
                st.success("🎉 Model retrained successfully!")
                
                # Show improvement details
                if learning_manager.performance_history:
                    latest = learning_manager.performance_history[-1]
                    improvement = latest.get('improvement', 0)
                    new_accuracy = latest.get('new_accuracy', 0)
                    
                    if improvement > 0:
                        st.success(f"📈 Model improved by {improvement:.1%}! New accuracy: {new_accuracy:.1%}")
                    elif improvement < 0:
                        st.warning(f"📉 Model accuracy decreased by {abs(improvement):.1%}. This can happen with limited feedback.")
                    else:
                        st.info("📊 Model accuracy remained stable.")
                
                # Clear uncertain cases cache to refresh
                st.rerun()
            else:
                st.error("❌ Model retraining failed. Check logs for details.")
                
        except Exception as e:
            st.error(f"❌ Retraining error: {e}")

def enhanced_main_with_self_learning():
    """Entry point for the application when run from command line"""
    # Your main application logic here
    st.title("Bank Reconciliation AI - Main Application")
    st.write("Intelligent bank reconciliation with machine learning and self-learning capabilities.")

    # File upload sections
    st.header("📁 Upload Transaction Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        bank_file = st.file_uploader("Upload Bank Statement", type=["csv", "xlsx"])
        
        if bank_file is not None:
            # Read bank file
            if bank_file.name.endswith('.csv'):
                bank_df = pd.read_csv(bank_file)
            else:
                bank_df = pd.read_excel(bank_file)
            
            st.write("Bank Statement Data:")
            st.dataframe(bank_df.head(20))

    with col2:
        erp_file = st.file_uploader("Upload ERP Transactions", type=["csv", "xlsx"])
        
        if erp_file is not None:
            # Read ERP file
            if erp_file.name.endswith('.csv'):
                erp_df = pd.read_csv(erp_file)
            else:
                erp_df = pd.read_excel(erp_file)
                
            st.write("ERP Transaction Data:")
            st.dataframe(erp_df.head(20))

    # Self-Learning Section
    create_self_learning_section()


    # Model Training Section (PLACE YOUR SNIPPET HERE)
    st.header("🤖 Model Training")
    st.write("Upload labeled training data to improve model accuracy.")
    
    label_file = st.file_uploader(
        "Upload Labeled Training Data", 
        type=["csv", "xlsx"],
        help="CSV/Excel file with transaction pairs and match labels (0/1)"
    )
    
    if label_file is not None:
        # Read label file
        if label_file.name.endswith('.csv'):
            label_df = pd.read_csv(label_file)
        else:
            label_df = pd.read_excel(label_file)
        
        st.write("Training Data Preview:")
        st.dataframe(label_df.head(10))
        
        # Show expected format
        with st.expander("Expected Training Data Format"):
            st.write("""
            Your training data should contain these columns:
            - `bank_amount`, `erp_amount`: Transaction amounts
            - `bank_description`, `erp_description`: Transaction descriptions  
            - `bank_date`, `erp_date`: Transaction dates
            - `label`: Label (1 for match, 0 for no match)
            
            Example:
            | bank_amount | erp_amount | bank_description | erp_description | bank_date | erp_date | label |
            |-------------|------------|------------------|-----------------|-----------|----------|----------|
            | 100.00      | 100.00     | Payment to ABC   | ABC Invoice     | 2024-01-01| 2024-01-01| 1       |
            | 50.00       | 75.00      | Store Purchase   | Different Store | 2024-01-02| 2024-01-05| 0       |
            """)

    # Training button with enhanced functionality
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            if label_file is not None:
                with st.spinner("Training model... This may take a few minutes."):
                    try:
                        # Train the model with labelled data
                        model = train_model(label_df)
                        st.success("✅ Model trained successfully!")
                        
                        # Show training results if available
                        if hasattr(model, 'score') and hasattr(model, 'feature_importances_'):
                            st.subheader("Training Results")
                            
                            # Show feature importance if available
                            feature_names = ['amount_difference', 'date_difference', 
                                           'description_similarity', 'signed_amount_match', 'same_day']
                            importance_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': model.feature_importances_
                            }).sort_values('Importance', ascending=False)
                            
                            st.write("Feature Importance:")
                            st.bar_chart(importance_df.set_index('Feature'))
                        
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")
                        st.exception(e)
            else:
                st.warning("⚠️ Please upload labeled training data to train the model.")

    # Reconciliation Section (modified to use self-learning)
    st.header("🔄 Run Reconciliation")
    
    # Run reconciliation when both files are uploaded
    if bank_file is not None and erp_file is not None:
        
        # Configuration options
        with st.expander("⚙️ Advanced Settings"):
            confidence_threshold = st.slider(
                "Match Confidence Threshold", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.7, 
                step=0.05,
                help="Higher values = more conservative matching"
            )
            
            uncertainty_threshold = st.slider(
                "Uncertainty Review Threshold",
                min_value=0.0,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Cases with uncertainty above this will be flagged for review"
            )

            max_combinations = st.number_input(
                "Max Combinations to Process", 
                min_value=1000, 
                max_value=100000, 
                value=50000,
                help="Limit processing for large datasets"
            )
        
        if st.button("🔍 Run Reconciliation", type="primary", use_container_width=True):
            
            with st.spinner("Processing reconciliation with AI enhancement..."):
                try:
                    # Update config if user changed settings
                    import config
                    config.MATCH_CONFIDENCE_THRESHOLD = confidence_threshold
                    
                    # Run enhanced reconciliation with self-learning
                    from reconcile import run_reconciliation_with_learning
                    matched_report, unmatched, summary_df, uncertain_cases = run_reconciliation_with_learning(bank_df, erp_df)
                    
                    # Store results in session state for self-learning features
                    st.session_state.last_results = {
                        'matched_report': matched_report,
                        'unmatched': unmatched,
                        'summary_df': summary_df,
                        'uncertain_cases': uncertain_cases,
                        'bank_df': bank_df,
                        'erp_df': erp_df
                    }
                    
                    # Display summary
                    st.success("✅ Smart reconciliation completed successfully!")
                    
                    # Enhanced summary with self-learning metrics
                    st.header("📊 Enhanced Reconciliation Summary")
                    
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            label="Matches Found",
                            value=len(matched_report)
                        )
                    
                    with col2:
                        match_rate = (len(matched_report) / len(bank_df)) * 100 if len(bank_df) > 0 else 0
                        st.metric(
                            label="Match Rate",
                            value=f"{match_rate:.1f}%"
                        )
                    
                    with col3:
                        st.metric(
                            label="Uncertain Cases",
                            value=len(uncertain_cases),
                            help="Cases that need human review"
                        )
                    
                    with col4:
                        avg_confidence = matched_report['match_confidence'].mean() if not matched_report.empty else 0
                        st.metric(
                            label="Avg Confidence",
                            value=f"{avg_confidence:.3f}"
                        )
                    
                    # Summary table
                    st.subheader("Detailed Summary")
                    st.table(summary_df)
                    
                    # Detailed results tabs
                    tab1, tab2, tab3, tab4 = st.tabs(["✅ Matched", "❌ Unmatched", "📊 Summary", "🔍 Uncertain"])
                    
                    with tab1:
                        if not matched_report.empty:
                            st.write(f"Found {len(matched_report)} matched transactions:")
                            
                            # Color code by confidence
                            def color_confidence(val):
                                if val >= 0.9:
                                    return 'background-color: #d4edda'  # Green
                                elif val >= 0.7:
                                    return 'background-color: #fff3cd'  # Yellow
                                else:
                                    return 'background-color: #f8d7da'  # Red
                            
                            styled_df = matched_report.style.applymap(
                                color_confidence, 
                                subset=['match_confidence'] if 'match_confidence' in matched_report.columns else []
                            )
                            
                            st.dataframe(styled_df, use_container_width=True)
                            
                            # Download button
                            csv = matched_report.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Matched Transactions",
                                data=csv,
                                file_name="matched_transactions.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("No matches found.")
                    
                    with tab2:
                        if not unmatched.empty:
                            st.write(f"Found {len(unmatched)} unmatched transactions:")
                            st.dataframe(unmatched, use_container_width=True)
                            
                            # Highlight anomalies
                            if 'anomaly' in unmatched.columns:
                                anomalies = unmatched[unmatched['anomaly'] == -1]
                                if not anomalies.empty:
                                    st.warning(f"⚠️ {len(anomalies)} potential anomalies detected:")
                                    st.dataframe(anomalies, use_container_width=True)
                        else:
                            st.success("🎉 All transactions matched!")
                    
                    with tab3:
                        st.subheader("Detailed Summary")
                        st.table(summary_df)
                        
                        # AI insights
                        st.subheader("💡 AI Insights")
                        if len(uncertain_cases) > 0:
                            st.info(f"🔍 {len(uncertain_cases)} cases need human review to improve model accuracy.")
                        
                        if match_rate > 90:
                            st.success("🎯 Excellent match rate! Your data quality is very good.")
                        elif match_rate > 70:
                            st.warning("📈 Good match rate. Consider reviewing uncertain cases to improve accuracy.")
                        else:
                            st.error("📉 Low match rate. Data quality issues detected. Review unmatched transactions.")
                    
                    with tab4:
                        if uncertain_cases:
                            st.write(f"Found {len(uncertain_cases)} cases needing review:")
                            st.info("💡 Reviewing these cases will help improve the AI model!")
                            
                            # Show top uncertain cases
                            for case in uncertain_cases[:5]:
                                st.write(f"• Case {case['index']}: {case['reason']} (Confidence: {case['confidence']:.3f})")
                            
                            if len(uncertain_cases) > 5:
                                st.write(f"... and {len(uncertain_cases) - 5} more cases.")
                            
                            st.write("👆 Go to the **Self-Learning** section above to review these cases!")
                        else:
                            st.success("🎉 No uncertain cases! The model is confident in all predictions.")
                
                except Exception as e:
                    st.error(f"❌ An error occurred during reconciliation: {str(e)}")
                    st.exception(e)
    else:
        st.info("👆 Please upload both bank and ERP transaction files to proceed.")
  
    
if __name__ == "__main__":
    enhanced_main_with_self_learning()