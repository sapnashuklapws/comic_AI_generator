import streamlit as st
import os
import io
import time
from PIL import Image
import google.generativeai as genai

class ComicGenerator:
    def __init__(self):
        # Configure Google GenAI with Streamlit secrets
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        self.text_model_name = "gemini-2.5-flash"
        # Imagen 3 is Google's state-of-the-art image generation model
        self.image_model_name = "imagen-3.0-generate-002"
        
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
        
        model = genai.GenerativeModel(self.text_model_name)
        response = model.generate_content(prompt)
        return response.text

    def generate_image_prompts(self, story, child_image):
        prompt = f"""Given this story for a children's comic book:
        {story}
        
        And considering we have a photo of a child who will be the main character,
        generate 6-8 detailed image prompts that will work well with Imagen 3.
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
        
        model = genai.GenerativeModel(self.text_model_name)
        response = model.generate_content(prompt)
        content = response.text.strip()
        
        # Split into panels and clean them up
        panels = []
        if '🎨' in content:
            panels = [p.strip() for p in content.split('🎨') if p.strip()]
        else:
            panels = [p.strip() for p in content.split('Panel') if p.strip()]
        
        cleaned_panels = []
        for panel in panels:
            if panel.startswith((':', '1:', '2:', '3:', '4:', '5:', '6:', '7:', '8:', '9:', '10:')):
                panel = panel.split(':', 1)[1]
            cleaned_panels.append(panel.strip())
        
        return cleaned_panels

    def generate_comic_panel(self, prompt):
        max_retries = 3
        retry_delay = 2  # seconds
        
        cleaned_prompt = str(prompt).replace('"', '').strip()
        cleaned_prompt = cleaned_prompt.split(':', 1)[-1].strip() if ':' in cleaned_prompt else cleaned_prompt
        
        # Append target style variables directly to the prompt for Google Imagen
        final_prompt = f"{cleaned_prompt}, child-friendly illustrated look, children's book illustration style, bright colors, soft lighting, vibrant"
        
        for attempt in range(max_retries):
            try:
                # Call Google's Imagen Model
                model = genai.GenerativeModel(self.image_model_name)
                result = model.generate_content(
                    final_prompt,
                    generation_config={
                        "response_mime_type": "image/jpeg"
                    }
                )
                
                # Verify and convert bytes to a PIL Image
                for part in result.candidates[0].content.parts:
                    if part.inline_data:
                        image_bytes = part.inline_data.data
                        image = Image.open(io.BytesIO(image_bytes))
                        return image
                
                return None
                    
            except Exception as e:
                st.error(f"Error generating image on attempt {attempt + 1}: {str(e)}")
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
        
        model = genai.GenerativeModel(self.text_model_name)
        response = model.generate_content(prompt)
        return response.text

def main():
    st.title("Comic Book Generator")
    st.write("Create personalized educational comics")
    
    generator = ComicGenerator()
    
    theme = st.text_input("Enter an educational theme (e.g., 'Learning to Share', 'Exploring Nature')")
    uploaded_file = st.file_uploader("Upload a photo of the child", type=['png', 'jpg', 'jpeg'])
    
    if theme and uploaded_file:
        if st.button("Generate Story Options"):
            with st.spinner("Generating story options..."):
                story_options = generator.generate_story_options(theme)
                st.session_state.story_options = story_options
                st.session_state.story_text = story_options
        
        if 'story_options' in st.session_state:
            with st.container():
                st.markdown("## ✨ Story Options")
                
                story_text = st.session_state.story_text
                cleaned_text = str(story_text).strip()
                stories = [s.strip() for s in cleaned_text.split('🌟') if s.strip()]
                
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
        
            st.markdown("### 📝 Select Your Story")
            selected_story = st.text_area(
                "Copy and paste your preferred story here:",
                height=150,
                help="Paste the complete story option you chose from above",
                key="story_selector"
            )
            
            if selected_story and st.button("Elaborate Story", key="generate_panels"):
                with st.spinner("Elaborating story..."):
                    elaborated_story = generator.elaborate_story(selected_story)
                    cleaned_story = str(elaborated_story).strip()
                    
                    st.session_state.elaborated_story_raw = elaborated_story
                    st.session_state.elaborated_story_clean = cleaned_story
                    
                    st.markdown("## 📖 Elaborated Story")
                    sections = cleaned_story.split("\n\n")
                    for section in sections:
                        if section.strip():
                            st.markdown(section.strip())
            
            elif 'elaborated_story_clean' in st.session_state:
                st.markdown("## 📖 Elaborated Story")
                sections = st.session_state.elaborated_story_clean.split("\n\n")
                for section in sections:
                    if section.strip():
                        st.markdown(section.strip())
        
            if st.button("Generate Comic Panels with this Story", key="confirm_panels"):
                if 'elaborated_story_clean' not in st.session_state:
                    st.error("Please elaborate the story first before generating panels.")
                    return
                
                with st.spinner("Generating image prompts..."):
                    elaborated_story = st.session_state.elaborated_story_clean
                    image_prompts = generator.generate_image_prompts(elaborated_story, uploaded_file)
                    st.session_state.image_prompts = image_prompts
                    
                    st.markdown("## 🎨 Comic Panels")
                    cols = st.columns(2)
                    
                    for i, prompt in enumerate(image_prompts):
                        with cols[i % 2]:
                            with st.container():
                                st.markdown(f"### 🖼️ Panel {i+1}")
                                display_prompt = str(prompt).strip()
                                
                                if 'cleaned_prompts' not in st.session_state:
                                    st.session_state.cleaned_prompts = {}
                                st.session_state.cleaned_prompts[i] = display_prompt
                                
                                with st.expander("✨ View prompt details", expanded=False):
                                    st.markdown(f"**Scene Description:**\n{display_prompt}")
                                
                                with st.spinner(f"Creating panel {i+1}..."):
                                    panel_image = generator.generate_comic_panel(prompt)
                                    if panel_image:
                                        st.image(panel_image, use_column_width=True)
                                        if 'generated_images' not in st.session_state:
                                            st.session_state.generated_images = {}
                                        st.session_state.generated_images[i] = panel_image
                                    else:
                                        st.error(f"Could not generate panel {i+1}. Please try again.")
        
        elif 'image_prompts' in st.session_state:
            st.markdown("## 🎨 Comic Panels")
            cols = st.columns(2)
            
            for i, prompt in enumerate(st.session_state.image_prompts):
                with cols[i % 2]:
                    with st.container():
                        st.markdown(f"### 🖼️ Panel {i+1}")
                        
                        if 'cleaned_prompts' in st.session_state and i in st.session_state.cleaned_prompts:
                            with st.expander("✨ View prompt details", expanded=False):
                                st.markdown(f"**Scene Description:**\n{st.session_state.cleaned_prompts[i]}")
                        
                        if 'generated_images' in st.session_state and i in st.session_state.generated_images:
                            st.image(st.session_state.generated_images[i], use_column_width=True)

if __name__ == "__main__":
    st.set_page_config(page_title="Comic Generator", layout="wide")
    main()
