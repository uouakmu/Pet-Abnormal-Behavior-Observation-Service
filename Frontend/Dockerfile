# Stage 1: Build the Flutter Web application
FROM ghcr.io/cirruslabs/flutter:stable AS build

# Set working directory
WORKDIR /app

# Copy pubspec files first to leverage Docker cache
COPY pubspec.yaml pubspec.lock ./

# Fetch dependencies
RUN flutter pub get

# Copy the rest of the application code
COPY . .

# Build the web version of the application
# (It might take some time depending on project size)
RUN flutter build web

# Stage 2: Serve the application using Nginx
FROM nginx:alpine

# Copy the Nginx configuration if you have custom one (optional)
# COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy the built web files from the previous stage to Nginx's HTML content directory
COPY --from=build /app/build/web /usr/share/nginx/html

# Expose port (default is 80)
EXPOSE 80

# Start Nginx server
CMD ["nginx", "-g", "daemon off;"]
