import streamlit as st
import replicate
import os
import requests
from PIL import Image
from io import BytesIO
import anthropic
import time

class ComicGenerator:
    def __init__(self):
        # Use environment variable or Streamlit secrets for both APIs
        self.client = anthropic.Client(api_key=st.secrets["ANTHROPIC_API_KEY"])
        # Initialize Replicate client
        self.replicate_client = replicate.Client(api_token=st.secrets["REPLICATE_API_KEY"])
        
    def generate_story_options(self, theme):
        prompt = f"""Answer in the same language as the user's input.
        As an expert in education and children's literature, generate 3 different story options for a children's comic book about {theme}. 
        Each story should:
        - Be suitable for children aged 4-8
        - Incorporate educational principles like independence, natural learning, and respect
        - Have a clear moral or educational message
        - Be structured in 6-8 scenes
        - Include interactive elements or questions
        - Answer directly with the story options, without any other text
        - You must generate 10 different options. if the user specifies a number, you must generate that number of story options.
        
        Format your response EXACTLY like this example (keep the emojis):

        🌟 Story Option 1: The Garden Adventure
        📖 Description: Sarah learns about plant growth and responsibility.
        🎯 Moral: Taking care of living things teaches us patience and love.

        🌟 Story Option 2: The Sharing Circle
        📖 Description: Tom discovers the joy of sharing with friends.
        🎯 Moral: Sharing brings happiness to ourselves and others.

        🌟 Story Option 3: The Clean-Up Hero
        📖 Description: Maria learns to organize her room independently.
        🎯 Moral: Being organized helps us be more independent."""
        
        message = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content

    def generate_image_prompts(self, story, child_image):
        prompt = f"""Given this story for a children's comic book:
        {story}
        
        And considering we have a photo of a child who will be the main character,
        generate 6-8 detailed image prompts that will work well with Stable Diffusion.
        Each prompt should:
        - Describe a key scene from the story
        - Include style directions for a child-friendly, illustrated look
        - Mention it should be in the style of a children's book illustration
        - Be safe and appropriate for children
        - You must generate 10 different options. if the user specifies a number, you must generate that number of story options.
        
        Format your response EXACTLY like this example:
        
        🎨 Panel 1:
        [Your detailed prompt here]
        
        🎨 Panel 2:
        [Your detailed prompt here]
        
        (and so on...)"""
        
        message = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Handle the response content
        content = message.content
        if isinstance(content, (list, dict)):
            content = str(content)
        
        # Clean up the text
        content = content.replace("[TextBlock(text='", "").replace("', type='text')]", "")
        content = content.replace('\\n', '\n').strip()
        
        # Split into panels and clean them up
        panels = []
        if '🎨' in content:
            panels = [p.strip() for p in content.split('🎨') if p.strip()]
        else:
            # Fallback: try to split by "Panel" if no emoji is present
            panels = [p.strip() for p in content.split('Panel') if p.strip()]
        
        # Clean up panel format
        cleaned_panels = []
        for panel in panels:
            # Remove panel number if present at the start
            if panel.startswith((':',  '1:', '2:', '3:', '4:', '5:', '6:', '7:', '8:', '9:', '10:')):
                panel = panel.split(':', 1)[1]
            cleaned_panels.append(panel.strip())
        
        return cleaned_panels

    def generate_comic_panel(self, prompt):
        max_retries = 3
        retry_delay = 2  # seconds
        
        # Clean the prompt
        if isinstance(prompt, str):
            cleaned_prompt = prompt
        else:
            cleaned_prompt = getattr(prompt, 'text', str(prompt))
            
        cleaned_prompt = (cleaned_prompt.replace("[TextBlock(text='", "")
                                  .replace("', type='text')]", "")
                                  .replace('"', '')
                                  .strip())
        cleaned_prompt = cleaned_prompt.split(':', 1)[-1].strip() if ':' in cleaned_prompt else cleaned_prompt
        
        for attempt in range(max_retries):
            try:
                # Check internet connection
                requests.get("https://api.replicate.com/v1/predictions", timeout=5)
                
                # Generate the image
                output = self.replicate_client.run(
                    "lucataco/sdxl-lcm:fbbd475b1084de80c47c35bfe4ae64b964294aa7e237e6537eed938cfd24903d",
                    input={
                        "prompt": cleaned_prompt,
                        "negative_prompt": "scary, violent, inappropriate, realistic, photographic",
                        "width": 768,
                        "height": 512,
                        "scheduler": "KarrasDPM",
                        "num_inference_steps": 8,
                        "guidance_scale": 7.5
                    }
                )
                
                # Handle the output
                if output and isinstance(output, list) and len(output) > 0:
                    # Download the image and convert to PIL Image
                    response = requests.get(output[0])
                    if response.status_code == 200:
                        image = Image.open(BytesIO(response.content))
                        return image
                return None
                
            except requests.exceptions.RequestException as e:
                st.warning(f"Network error on attempt {attempt + 1}/{max_retries}. Retrying...")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    st.error(f"Network error after {max_retries} attempts: {str(e)}")
                    return None
                    
            except Exception as e:
                st.error(f"Error generating image: {str(e)}")
                if attempt < max_retries - 1:
                    st.info(f"Retrying... Attempt {attempt + 2}/{max_retries}")
                    time.sleep(retry_delay)
                else:
                    return None

    def elaborate_story(self, story_summary):
        prompt = f"""Based on this story summary:
        {story_summary}
        
        Create a detailed narrative script for a children's comic book that:
        - Expands the story into 6-8 scenes
        - Includes dialogue and narration for each scene
        - Maintains an educational and engaging tone for children aged 4-8
        - Incorporates interactive questions or reflection points
        - Follows educational principles
        
        Format your response EXACTLY like this:
        
        📖 Title: [Story Title]
        
        🎬 Scene 1:
        Narration: [Scene description]
        Dialogue: [Character dialogue]
        💭 Interactive Moment: [Question or activity for the reader]
        
        🎬 Scene 2:
        [Continue format for each scene...]"""
        
        message = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content

def main():
    st.title("Comic Book Generator")
    st.write("Create personalized educational comics")
    
    # Initialize the generator
    generator = ComicGenerator()
    
    # Theme input
    theme = st.text_input("Enter an educational theme (e.g., 'Learning to Share', 'Exploring Nature')")
    
    # Image upload
    uploaded_file = st.file_uploader("Upload a photo of the child", type=['png', 'jpg', 'jpeg'])
    
    if theme and uploaded_file:
        # Generate story options
        if st.button("Generate Story Options"):
            with st.spinner("Generating story options..."):
                story_options = generator.generate_story_options(theme)
                # Save story options to session state
                st.session_state.story_options = story_options
                st.session_state.story_text = story_options  # Save the raw text as well
        
        # Display story options if they exist in session state
        if 'story_options' in st.session_state:
            with st.container():
                st.markdown("## ✨ Story Options")
                
                # Use the saved story text
                story_text = st.session_state.story_text
                
                # Clean and extract the actual text content
                if isinstance(story_text, str):
                    cleaned_text = story_text
                else:
                    cleaned_text = getattr(story_text, 'text', str(story_text))
                
                # Remove TextBlock wrapper if present
                cleaned_text = cleaned_text.replace("[TextBlock(text='", "").replace("', type='text')]", "")
                cleaned_text = cleaned_text.replace('\\n', '\n').strip()
                stories = [s.strip() for s in cleaned_text.split('🌟') if s.strip()]
                
                # Display each story in a nice format
                for story in stories:
                    lines = [line.strip() for line in story.split('\n') if line.strip()]
                    
                    if not lines:
                        continue
                    
                    title = next((line for line in lines if "historia" in line.lower() or "story" in line.lower()), lines[0])
                    
                    with st.expander(f"🌟 {title}", expanded=True):
                        for line in lines:
                            if line != title:
                                if '📖' in line:
                                    st.markdown(f"**{line}**")
                                elif '🎯' in line:
                                    st.markdown(f"_{line}_")
                                else:
                                    st.markdown(line)
        
            # Story selection
            st.markdown("### 📝 Select Your Story")
            selected_story = st.text_area(
                "Copy and paste your preferred story here:",
                height=150,
                help="Paste the complete story option you chose from above",
                key="story_selector"
            )
            
            if selected_story and st.button("Elaborate Story", key="generate_panels"):
                with st.spinner("Elaborating story..."):
                    # First, elaborate the story
                    elaborated_story = generator.elaborate_story(selected_story)
                    
                    # Clean up the story text
                    if isinstance(elaborated_story, (str, bytes)):
                        cleaned_story = elaborated_story
                    else:
                        cleaned_story = str(elaborated_story)
                    
                    # More thorough cleaning
                    cleaned_story = (cleaned_story.replace("TextBlock(text='", "")
                                               .replace("text='", "")
                                               .replace("')", "")
                                               .replace("',", "")
                                               .replace(" type='text')", "")
                                               .replace("[", "")
                                               .replace("]", "")
                                               .replace("\\n", "\n")
                                               .strip())
                    
                    # Save both raw and cleaned versions to session state
                    st.session_state.elaborated_story_raw = elaborated_story
                    st.session_state.elaborated_story_clean = cleaned_story
                    
                    # Display the elaborated story with proper formatting
                    st.markdown("## 📖 Elaborated Story")
                    # Split the story into sections and format each properly
                    sections = cleaned_story.split("\n\n")
                    for section in sections:
                        if section.strip():
                            st.markdown(section.strip())
            
            # Add this section to display the saved story if it exists
            elif 'elaborated_story_clean' in st.session_state:
                st.markdown("## 📖 Elaborated Story")
                sections = st.session_state.elaborated_story_clean.split("\n\n")
                for section in sections:
                    if section.strip():
                        st.markdown(section.strip())
        
            # Add a confirmation button to proceed with comic generation
            if st.button("Generate Comic Panels with this Story", key="confirm_panels"):
                if 'elaborated_story_clean' not in st.session_state:
                    st.error("Please elaborate the story first before generating panels.")
                    return
                
                with st.spinner("Generating image prompts..."):
                    # Use the saved elaborated story from session state
                    elaborated_story = st.session_state.elaborated_story_clean
                    
                    # Continue with the existing image generation logic
                    image_prompts = generator.generate_image_prompts(elaborated_story, uploaded_file)
                    st.session_state.image_prompts = image_prompts
                    
                    st.markdown("## 🎨 Comic Panels")
                    cols = st.columns(2)
                    
                    for i, prompt in enumerate(image_prompts):
                        with cols[i % 2]:
                            with st.container():
                                st.markdown(f"### 🖼️ Panel {i+1}")
                                
                                # Clean up the prompt text for display
                                if isinstance(prompt, (str, bytes)):
                                    display_prompt = prompt
                                else:
                                    display_prompt = str(prompt)
                                
                                # Remove all TextBlock artifacts and clean up
                                display_prompt = (display_prompt.replace("[TextBlock(text=", "")
                                                            .replace("text=", "")
                                                            .replace("[TextBlock(", "")
                                                            .replace("type='text')]", "")
                                                            .replace("'", "")
                                                            .replace('"', "")
                                                            .strip())
                                
                                # Save cleaned prompt to session state
                                if 'cleaned_prompts' not in st.session_state:
                                    st.session_state.cleaned_prompts = {}
                                st.session_state.cleaned_prompts[i] = display_prompt
                                
                                # Display the prompt in a cleaner format
                                with st.expander("✨ View prompt details", expanded=False):
                                    st.markdown(f"""
                                    **Scene Description:**
                                    {display_prompt}
                                    """)
                                
                                # Generate and display the image
                                with st.spinner(f"Creating panel {i+1}..."):
                                    image_url = generator.generate_comic_panel(prompt)
                                    if image_url:
                                        st.image(image_url, use_column_width=True)
                                        # Save image URL to session state
                                        if 'generated_images' not in st.session_state:
                                            st.session_state.generated_images = {}
                                        st.session_state.generated_images[i] = image_url
                                    else:
                                        st.error(f"Could not generate panel {i+1}. Please try again.")
        
        # Display saved panels if they exist
        elif 'image_prompts' in st.session_state:
            st.markdown("## 🎨 Comic Panels")
            cols = st.columns(2)
            
            for i, prompt in enumerate(st.session_state.image_prompts):
                with cols[i % 2]:
                    with st.container():
                        st.markdown(f"### 🖼️ Panel {i+1}")
                        
                        # Display saved prompt
                        if 'cleaned_prompts' in st.session_state and i in st.session_state.cleaned_prompts:
                            with st.expander("✨ View prompt details", expanded=False):
                                st.markdown("""
                                **Scene Description:**
                                {}
                                """.format(st.session_state.cleaned_prompts[i]))
                        
                        # Display saved image
                        if 'generated_images' in st.session_state and i in st.session_state.generated_images:
                            st.image(st.session_state.generated_images[i], use_column_width=True)

if __name__ == "__main__":
    st.set_page_config(page_title="Comic Generator", layout="wide")
    main()