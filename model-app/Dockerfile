FROM public.ecr.aws/lambda/python:3.8

# Copy function code
COPY ./model-app/model-app.py ${LAMBDA_TASK_ROOT}

# Avoid cache purge by adding requirements first
COPY ./model-app/requirements.txt ${LAMBDA_TASK_ROOT}

RUN pip install --no-cache-dir -r requirements.txt


# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "model-app.handler"]


