# Dockerfile (Frontend)
FROM nginx:alpine

# Copy frontend files to nginx html directory
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY . /usr/share/nginx/html

# Expose nginx port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
