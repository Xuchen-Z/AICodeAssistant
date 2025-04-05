from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import feedback as fb
from debugger import process_code
#from rag_debugger import process_code
from summarizer import process_summary
from threading import Thread


app = Flask(__name__, template_folder='templates')
CORS(app)  # This allows frontend to make requests to this backend

@app.route('/')
def index():
    return render_template('index.html')  # This renders the HTML file

@app.route('/api/analyze', methods=['POST'])
def analyze_code():
    try:
        if 'codeFile' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['codeFile']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        try:
            code_content = file.read().decode('utf-8')
        except Exception as e:
            return jsonify({'error': 'Error reading file content'}), 400

        try:
            result = process_code(code_content)
            code_summary = process_summary(code_content)  # Call summarizer

            if not code_summary:
                code_summary = "Summary not available."

            return jsonify({'result': result, 'code_summary': code_summary})

        except Exception as e:
            return jsonify({'error': f'AI processing error: {str(e)}'}), 500

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    try:
        feedback_text = request.form.get('feedback', '').strip()
        if feedback_text:
            fb.init_feedback_log()
            new_feedback = fb.process_feedback(feedback_text)
            fb.save_feedback(new_feedback)
            return jsonify({'message': 'Feedback submitted successfully!'}), 200
        else:
            return jsonify({'error': 'No feedback provided'}), 400
    except Exception as e:
        return jsonify({'error': f'Error processing feedback: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5017)