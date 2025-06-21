# Nireon V4 Proto-Plane - Math Dialect Image
# This image builds upon the base image to add math-specific libraries.

# Layer on top of the secure base image.
FROM nireon-proto-base:latest

# Install additional math-specific packages required for the 'math' eidos.
# These packages are validated against the allowlist in the ProtoValidator.
RUN pip install --no-cache-dir \
    sympy==1.12 \
    statsmodels==0.14.1 \
    matplotlib==3.8.4 \
    seaborn==0.13.2

# Optional: Environment variable for introspection inside the container.
ENV PROTO_DIALECT=math