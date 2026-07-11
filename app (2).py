import streamlit as st
import os
import io
import time
from PIL import Image
# Modern Google GenAI SDK imports
from google import genai
from google.genai import types

class ComicGenerator:
    def __init__(self):
        # Configure Google GenAI with Streamlit secrets
        if "GOOGLE_API_KEY" not in st.secrets:
            st.error("Missing 'GOOGLE_API_KEY' in Streamlit Secrets. Please configure it in app settings.")
            st.stop()
            
        # Initialize the modern SDK Client
        self.client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
        self.text_model_name = "gemini-2.5-flash"
        
        # FIX: Changed to the correct Gemini Developer API identifier for Imagen 3
        self.image_model_name = "gemini-2.5-flash-image"
        
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
        
        Format your response EXACTLY like this example (keep the emojis):

        🌟 Story Option 1: The Garden Adventure
        📖 Description: Sarah learns about plant growth and responsibility.
        🎯 Moral: Taking care of living things teaches us patience and love.

        🌟 Story Option 2: The Sharing Circle
        📖 Description: Tom discovers the joy of sharing with friends.
        🎯 Moral: Sharing brings happiness to ourselves and others."""
        
        response = self.client.models.generate_content(
            model=self.text_model_name,
            contents=prompt,
        )
        return response.text

    def generate_image_prompts(self, story):
        prompt = f"""Given this story for a children's comic book:
        {story}
        
        Generate 6-8 detailed image prompts that will work well with Imagen 3.
        Each prompt should:
        - Describe a key scene from the story
        - Include style directions for a child-friendly, illustrated look
        - Mention it should be in the style of a children's book illustration
        - Be safe and appropriate for children
        
        Format your response EXACTLY like this example:
        
        🎨 Panel 1:
        [Your detailed prompt here]
        
        🎨 Panel 2:
        [Your detailed prompt here]"""
        
        response = self.client.models.generate_content(
            model=self.text_model_name,
            contents=prompt,
        )
        content = response.text.strip()
        
        # Split into panels and clean them up
        if '🎨' in content:
            panels = [p.strip() for p in content.split('🎨') if p.strip()]
        else:
            panels = [p.strip() for p in content.split('Panel') if p.strip()]
        
        cleaned_panels = []
        for panel in panels:
            if ':' in panel:
                panel = panel.split(':', 1)[1]
            cleaned_panels.append(panel.strip())
        
        return cleaned_panels

    def generate_comic_panel(self, prompt):
        max_retries = 3
        retry_delay = 2  # seconds
        
        cleaned_prompt = str(prompt).replace('"', '').strip()
        if ':' in cleaned_prompt:
            cleaned_prompt = cleaned_prompt.split(':', 1)[-1].strip()
            
        final_prompt = f"{cleaned_prompt}, child-friendly illustrated look, children's book illustration style, bright colors, soft lighting, vibrant"
        
        for attempt in range(max_retries):
            try:
                # API Call using correct model string and configs
                result = self.client.models.generate_images(
                    model=self.image_model_name,
                    prompt=final_prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        output_mime_type="image/jpeg",
                        # FIX: Changed from "3:2" to "4:3" (supported values are: "1:1", "3:4", "4:3", "9:16", "16:9")
                        aspect_ratio="4:3" 
                    )
                )
                
                # Unpack response image bytes
                for generated_image in result.generated_images:
                    return Image.open(io.BytesIO(generated_image.image.image_bytes))
                
                return None
                    
            except Exception as e:
                st.error(f"Error generating image on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
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
        
        Format your response EXACTLY like this:
        
        📖 Title: [Story Title]
        
        🎬 Scene 1:
        Narration: [Scene description]
        Dialogue: [Character dialogue]
        💭 Interactive Moment: [Question or activity for the reader]"""
        
        response = self.client.models.generate_content(
            model=self.text_model_name,
            contents=prompt,
        )
        return response.text

def main():
    st.title("Comic Book Generator")
    st.write("Create personalized educational comics using Gemini & Imagen 3")
    
    generator = ComicGenerator()
    theme = st.text_input("Enter an educational theme (e.g., 'Learning to Share', 'Exploring Nature')")
    
    if theme:
        if st.button("Generate Story Options"):
            with st.spinner("Generating story options..."):
                story_options = generator.generate_story_options(theme)
                st.session_state.story_options = story_options
                st.session_state.story_text = story_options
        
        if 'story_options' in st.session_state:
            st.markdown("## ✨ Story Options")
            stories = [s.strip() for s in st.session_state.story_text.split('🌟') if s.strip()]
            
            for story in stories:
                lines = [line.strip() for line in story.split('\n') if line.strip()]
                if not lines:
                    continue
                with st.expander(f"🌟 {lines[0]}", expanded=True):
                    for line in lines[1:]:
                        st.markdown(line)
        
            st.markdown("### 📝 Select Your Story")
            selected_story = st.text_area(
                "Copy and paste your preferred story here:",
                height=150,
                key="story_selector"
            )
            
            if selected_story and st.button("Elaborate Story", key="generate_panels"):
                with st.spinner("Elaborating story..."):
                    elaborated_story = generator.elaborate_story(selected_story)
                    st.session_state.elaborated_story_clean = elaborated_story.strip()
            
            if 'elaborated_story_clean' in st.session_state:
                st.markdown("## 📖 Elaborated Story")
                st.write(st.session_state.elaborated_story_clean)
        
                if st.button("Generate Comic Panels with this Story", key="confirm_panels"):
                    with st.spinner("Generating image prompts..."):
                        image_prompts = generator.generate_image_prompts(st.session_state.elaborated_story_clean)
                        st.session_state.image_prompts = image_prompts
                        
                        # Generate and cache panels into state to prevent them from wiping out on reload
                        st.session_state.generated_panels = []
                        for prompt in image_prompts:
                            panel_image = generator.generate_comic_panel(prompt)
                            st.session_state.generated_panels.append(panel_image)

            # Persistent panel generation block
            if 'generated_panels' in st.session_state and st.session_state.generated_panels:
                st.markdown("## 🎨 Comic Panels")
                cols = st.columns(2)
                for i, panel_image in enumerate(st.session_state.generated_panels):
                    with cols[i % 2]:
                        st.markdown(f"### 🖼️ Panel {i+1}")
                        if panel_image:
                            st.image(panel_image, use_container_width=True)
                        else:
                            st.error(f"Failed to load panel {i+1}")

if __name__ == "__main__":
    st.set_page_config(page_title="Comic Generator", layout="wide")
    main()
