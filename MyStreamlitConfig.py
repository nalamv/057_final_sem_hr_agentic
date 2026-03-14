CSS_STYLE="""
<style>
    /* User Message: Move to the right */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
        text-align: right;
        background-color: #dcf8c6; 
        border-radius: 15px;
        margin-left: auto;
        width: fit-content;
        max-width: 80%;
    }
    
    /* Assistant Message: Stay on the left */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background-color: #f0f0f0; 
        border-radius: 15px;
        margin-right: auto;
        width: fit-content;
        max-width: 80%;
    }
</style>
"""