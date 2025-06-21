# Nireon V4 Proto-Plane - Base Execution Image
# This provides a secure, minimal foundation for all Proto dialect runners.

# Use a slim, secure base image.
FROM python:3.12-slim

# Security: Create a non-root user for sandboxed code execution.
# This is a critical security measure to limit the blast radius of any potential issues.
RUN useradd -m -u 1000 protorunner

# Install a small set of common, trusted packages that are useful across many dialects.
# This pre-installation speeds up dialect-specific image builds.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scipy==1.13.0

# Create and set permissions for the app directory where code will be mounted.
WORKDIR /app
RUN chown protorunner:protorunner /app

# Switch to the non-root user for all subsequent commands.
USER protorunner

# Default entrypoint for the container.
# The -I flag runs Python in isolated mode, ignoring user-site packages and environment variables,
# further enhancing the security of the sandbox.
CMD ["python", "-I", "/app/execute.py"]