IDE: GitHub Codespaces
GitHub Codespaces
 is a cloud-hosted development environment built right into GitHub that gets you coding faster with pre-configured containers, adjustable compute power, and seamless integration with workflows like Actions and Copilot.
Why Codespaces helps
Reproducible onboarding
: Say goodbye to “works on my machine” woes—everyone uses the same setup for assignments or demos.
Anywhere access
: Jump back into your project from a laptop, tablet, or phone without having to reinstall anything.
Rapid experimentation & debugging
: Spin up short-lived environments on any branch, commit, or PR to isolate bugs or test features, or keep longer-lived codespaces for big projects.
Quick Setup
From the GitHub UI
Go to your repo and click 
Code → Codespaces → New codespace
.
Pick the branch and machine specs (2–32 cores, 8–64 GB RAM), then click 
Create codespace
.
In Visual Studio Code
Press 
Ctrl+Shift+P
 (or 
Cmd+Shift+P
 on Mac), choose 
Codespaces: Create New Codespace
, and follow the prompts.
Via GitHub CLI
gh auth login
gh codespace create 
--repo
 OWNER/REPO
gh codespace list    
# List all codespaces

gh codespace code    
# opens in your local VS Code

gh codespace 
ssh
     
# SSH into the codepsace
Copy to clipboard
Error
Copied
Features To Explore
Dev Containers
: Set up your environment the same way every time using a 
devcontainer.json
 or your own Dockerfile. 
Introduction to dev containers
Prebuilds
: Build bigger or more complex repos in advance so codespaces start up in a flash. 
About prebuilds
Port Forwarding
: Let Codespaces spot and forward the ports your web apps use automatically. 
Forward ports in Codespaces
Secrets & Variables
: Keep your environment variables safe in the Codespaces settings for your repo. 
Manage Codespaces secrets
Dotfiles Integration
: Bring in your dotfiles repo to customize shell settings, aliases, and tools in every codespace. 
Personalizing your codespaces
Machine Types & Cost Control
: Pick from VMs with 2 to 32 cores and track your usage in the billing dashboard. 
Managing Codespaces costs
VS Code & CLI Integration
: Flip between browser VS Code and your desktop editor, and script everything with the CLI. 
VS Code Remote: Codespaces
GitHub Actions
: Power up prebuilds and your CI/CD right inside codespaces using Actions workflows. 
Prebuilding your codespaces
Copilot in Codespaces
: Let Copilot help you write code with in-editor AI suggestions. 
Copilot in Codespaces














Previous




Containers: Docker, Podman












Next










Tunneling: ngrok





