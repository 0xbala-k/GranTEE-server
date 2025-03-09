# Deploying to Render.com

This guide will walk you through deploying the GranTEE Server application to Render.com.

## Prerequisites

1. A Render.com account
2. Your Google API key for Gemini
3. The GranTEE Server codebase

## Deployment Steps

### 1. Prepare Your Repository

Ensure your repository contains:
- All necessary code files
- A `requirements.txt` file with dependencies
- A `dockerfile` for containerized deployment
- The `.env.example` file (which will help you set up environment variables)

### 2. Create a New Web Service on Render

1. Log in to your Render.com dashboard
2. Click on "New +" and select "Web Service"
3. Connect your GitHub/GitLab repository containing the GranTEE Server code
4. Configure the following settings:
   - **Name**: Choose a name for your service (e.g., "grantee-server")
   - **Region**: Choose a region closest to your target users
   - **Branch**: Select the branch you want to deploy (usually `main` or `master`)
   - **Runtime**: Select "Docker"
   - **Instance Type**: Start with "Starter" for testing, scale as needed

### 3. Set Environment Variables

In the Render dashboard, add the following environment variables:
- `GOOGLE_API_KEY`: Your Google API key for accessing Gemini models
- Any other environment variables your application requires

### 4. Deploy Your Service

1. Click "Create Web Service"
2. Render will automatically build and deploy your Docker container
3. Wait for the build and deployment process to complete
4. Once deployed, you can access your API at the URL provided by Render

## Post-Deployment

### Testing Your Deployment

Once deployed, test your API using the examples in `sample_queries.md` to ensure everything is working correctly.

### Monitoring and Scaling

- Monitor your application's performance in the Render dashboard
- Scale your instance type if you need more resources
- Set up automatic scaling rules if needed for production use

### Setting Up a Custom Domain (Optional)

1. In the Render dashboard, navigate to your web service
2. Click on "Settings" and scroll to the "Custom Domain" section
3. Follow the instructions to add and configure your custom domain

## Troubleshooting

If you encounter issues with your deployment:

1. Check the build logs in the Render dashboard
2. Verify your environment variables are set correctly
3. Ensure your Dockerfile is properly configured
4. Check the application logs for runtime errors

For more detailed help, refer to the [Render.com documentation](https://render.com/docs). 