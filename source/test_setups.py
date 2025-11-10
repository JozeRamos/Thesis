import unittest
from Controller.controller import LLM
import os
import time

# LLM a testar:

# llama3-70b-8192
# gemma2-9b-it
# whisper-large-v3-turbo
# deepseek-r1-distill-llama-70b
# gpt-4o-mini

# Testes a fazer com os LLMs

# 1.	Com todos os elementos
# 2.	Sem RAG
# 3.	Sem Chat history
# 4.	Sem RAG & Chat history
# 5.	Sem Redirect
# 6.	Sem Next steps
# 7.	Sem CoT
# 8.	Sem Self Refine -> Sem RAG
# 9.	4 Rondas de Self Refine
# 10.	4 Rondas de Self Refine mas sem CoT

class TestController(unittest.TestCase):

    def setUp(self):
        # Initialize the controller before each test
        self.controller = LLM(os.path.join('source', 'Scenarios', 'calory_tracker.json'))

        self.controller.build_index()

        self.output_file = "source/tests/llm_setups_llama3-70b-8192.txt"
        
    
    def default_llm_setup(self, user_input):
        temp_text = self.controller.prepare_conversation_history()
        docs = self.controller.get_docs(user_input, 3)
        is_question = self.controller.is_question(user_input)

        # Step 2: Determine input type
        is_important = self.controller.is_important(user_input, is_question)
        is_info, final_prompt = self.controller.check_optional(user_input)


        if not is_question:
            completed, self.controller.current_stage = self.controller.check_stage_completion()


            if completed:
                return "End"
        
        if not is_info:
            if is_important:
                # Step 3: Handle important input
                cot_answer = self.controller.generate_cot_response(user_input, is_question, temp_text)

                middle_prompt = self.controller.self_consistency(user_input, cot_answer, 3, temp_text)
            else:
                # Step 4: Handle non-important input
                middle_prompt = self.controller.redirect_user(user_input, temp_text)

            # Step 5: Refine response
            feedback1 = self.controller.feedback(user_input, middle_prompt, temp_text, docs)
            refine1 = self.controller.refine(middle_prompt, user_input, feedback1, temp_text)

            feedback2 = self.controller.feedback(user_input, refine1, temp_text, docs)
            final_prompt = self.controller.refine(refine1, user_input, feedback2, temp_text)

        # Step 6: Finalize response
        if is_question or is_important or is_info:
            next_steps = self.controller.next_steps(user_input, final_prompt, temp_text)
            final_prompt = final_prompt + "\n\nNext Steps: " + next_steps

        # Step 7: Update conversation history
        self.controller.update_conversation_history(final_prompt, user_input)

        return final_prompt
    
    def no_RAG_llm_setup(self, user_input):
        temp_text = self.controller.prepare_conversation_history()
        docs = ""
        is_question = self.controller.is_question(user_input)

        # Step 2: Determine input type
        is_important = self.controller.is_important(user_input, is_question)
        is_info, final_prompt = self.controller.check_optional(user_input)


        if not is_question:
            completed, self.controller.current_stage = self.controller.check_stage_completion()


            if completed:
                return "End"
        
        if not is_info:
            if is_important:
                # Step 3: Handle important input
                cot_answer = self.controller.generate_cot_response(user_input, is_question, temp_text)

                middle_prompt = self.controller.self_consistency(user_input, cot_answer, 3, temp_text)
            else:
                # Step 4: Handle non-important input
                middle_prompt = self.controller.redirect_user(user_input, temp_text)

            # Step 5: Refine response
            feedback1 = self.controller.feedback(user_input, middle_prompt, temp_text, docs)
            refine1 = self.controller.refine(middle_prompt, user_input, feedback1, temp_text)

            feedback2 = self.controller.feedback(user_input, refine1, temp_text, docs)
            final_prompt = self.controller.refine(refine1, user_input, feedback2, temp_text)

        # Step 6: Finalize response
        if is_question or is_important or is_info:
            next_steps = self.controller.next_steps(user_input, final_prompt, temp_text)
            final_prompt = final_prompt + "\n\nNext Steps: " + next_steps

        # Step 7: Update conversation history
        self.controller.update_conversation_history(final_prompt, user_input)

        return final_prompt
    
    def no_chat_history_llm_setup(self, user_input):
        temp_text = self.controller.chat_history[0]
        docs = self.controller.get_docs(user_input, 3)
        is_question = self.controller.is_question(user_input)

        # Step 2: Determine input type
        is_important = self.controller.is_important(user_input, is_question)
        is_info, final_prompt = self.controller.check_optional(user_input)


        if not is_question:
            completed, self.controller.current_stage = self.controller.check_stage_completion()


            if completed:
                return "End"
        
        if not is_info:
            if is_important:
                # Step 3: Handle important input
                cot_answer = self.controller.generate_cot_response(user_input, is_question, temp_text)

                middle_prompt = self.controller.self_consistency(user_input, cot_answer, 3, temp_text)
            else:
                # Step 4: Handle non-important input
                middle_prompt = self.controller.redirect_user(user_input, temp_text)

            # Step 5: Refine response
            feedback1 = self.controller.feedback(user_input, middle_prompt, temp_text, docs)
            refine1 = self.controller.refine(middle_prompt, user_input, feedback1, temp_text)

            feedback2 = self.controller.feedback(user_input, refine1, temp_text, docs)
            final_prompt = self.controller.refine(refine1, user_input, feedback2, temp_text)

        # Step 6: Finalize response
        if is_question or is_important or is_info:
            next_steps = self.controller.next_steps(user_input, final_prompt, temp_text)
            final_prompt = final_prompt + "\n\nNext Steps: " + next_steps

        return final_prompt
    
    def no_rag_history_llm_setup(self, user_input):
        temp_text = self.controller.chat_history[0]
        docs = ""
        is_question = self.controller.is_question(user_input)

        # Step 2: Determine input type
        is_important = self.controller.is_important(user_input, is_question)
        is_info, final_prompt = self.controller.check_optional(user_input)


        if not is_question:
            completed, self.controller.current_stage = self.controller.check_stage_completion()


            if completed:
                return "End"
        
        if not is_info:
            if is_important:
                # Step 3: Handle important input
                cot_answer = self.controller.generate_cot_response(user_input, is_question, temp_text)

                middle_prompt = self.controller.self_consistency(user_input, cot_answer, 3, temp_text)
            else:
                # Step 4: Handle non-important input
                middle_prompt = self.controller.redirect_user(user_input, temp_text)

            # Step 5: Refine response
            feedback1 = self.controller.feedback(user_input, middle_prompt, temp_text, docs)
            refine1 = self.controller.refine(middle_prompt, user_input, feedback1, temp_text)

            feedback2 = self.controller.feedback(user_input, refine1, temp_text, docs)
            final_prompt = self.controller.refine(refine1, user_input, feedback2, temp_text)

        # Step 6: Finalize response
        if is_question or is_important or is_info:
            next_steps = self.controller.next_steps(user_input, final_prompt, temp_text)
            final_prompt = final_prompt + "\n\nNext Steps: " + next_steps

        return final_prompt
        
    
    def no_redirect(self, user_input):
        temp_text = self.controller.prepare_conversation_history()
        docs = self.controller.get_docs(user_input, 3)
        is_question = self.controller.is_question(user_input)

        # Step 2: Determine input type
        is_important = self.controller.is_important(user_input, is_question)
        is_info, final_prompt = self.controller.check_optional(user_input)


        if not is_question:
            completed, self.controller.current_stage = self.controller.check_stage_completion()


            if completed:
                return "End"
        
        if not is_info:
            # Step 3: Handle important input
            cot_answer = self.controller.generate_cot_response(user_input, is_question, temp_text)

            middle_prompt = self.controller.self_consistency(user_input, cot_answer, 3, temp_text)

            # Step 5: Refine response
            feedback1 = self.controller.feedback(user_input, middle_prompt, temp_text, docs)
            refine1 = self.controller.refine(middle_prompt, user_input, feedback1, temp_text)

            feedback2 = self.controller.feedback(user_input, refine1, temp_text, docs)
            final_prompt = self.controller.refine(refine1, user_input, feedback2, temp_text)

        # Step 6: Finalize response
        if is_question or is_important or is_info:
            next_steps = self.controller.next_steps(user_input, final_prompt, temp_text)
            final_prompt = final_prompt + "\n\nNext Steps: " + next_steps

        # Step 7: Update conversation history
        self.controller.update_conversation_history(final_prompt, user_input)

        return final_prompt
    
    def no_next_steps(self, user_input):
        temp_text = self.controller.prepare_conversation_history()
        docs = self.controller.get_docs(user_input, 3)
        is_question = self.controller.is_question(user_input)

        # Step 2: Determine input type
        is_important = self.controller.is_important(user_input, is_question)
        is_info, final_prompt = self.controller.check_optional(user_input)


        if not is_question:
            completed, self.controller.current_stage = self.controller.check_stage_completion()


            if completed:
                return "End"
        
        if not is_info:
            if is_important:
                # Step 3: Handle important input
                cot_answer = self.controller.generate_cot_response(user_input, is_question, temp_text)

                middle_prompt = self.controller.self_consistency(user_input, cot_answer, 3, temp_text)
            else:
                # Step 4: Handle non-important input
                middle_prompt = self.controller.redirect_user(user_input, temp_text)

            # Step 5: Refine response
            feedback1 = self.controller.feedback(user_input, middle_prompt, temp_text, docs)
            refine1 = self.controller.refine(middle_prompt, user_input, feedback1, temp_text)

            feedback2 = self.controller.feedback(user_input, refine1, temp_text, docs)
            final_prompt = self.controller.refine(refine1, user_input, feedback2, temp_text)


        # Step 7: Update conversation history
        self.controller.update_conversation_history(final_prompt, user_input)

        return final_prompt
    
    
    def no_CoT(self, user_input):
        temp_text = self.controller.prepare_conversation_history()
        docs = self.controller.get_docs(user_input, 3)
        is_question = self.controller.is_question(user_input)

        # Step 2: Determine input type
        is_important = self.controller.is_important(user_input, is_question)
        is_info, final_prompt = self.controller.check_optional(user_input)


        if not is_question:
            completed, self.controller.current_stage = self.controller.check_stage_completion()


            if completed:
                return "End"
        
        if not is_info:
            if is_important:
                middle_prompt = self.controller.self_consistency(user_input, user_input, 3, temp_text)
            else:
                # Step 4: Handle non-important input
                middle_prompt = self.controller.redirect_user(user_input, temp_text)

            # Step 5: Refine response
            feedback1 = self.controller.feedback(user_input, middle_prompt, temp_text, docs)
            refine1 = self.controller.refine(middle_prompt, user_input, feedback1, temp_text)

            feedback2 = self.controller.feedback(user_input, refine1, temp_text, docs)
            final_prompt = self.controller.refine(refine1, user_input, feedback2, temp_text)

        # Step 6: Finalize response
        if is_question or is_important or is_info:
            next_steps = self.controller.next_steps(user_input, final_prompt, temp_text)
            final_prompt = final_prompt + "\n\nNext Steps: " + next_steps

        # Step 7: Update conversation history
        self.controller.update_conversation_history(final_prompt, user_input)

        return final_prompt
    
    def no_self_refine(self, user_input):
        temp_text = self.controller.prepare_conversation_history()
        is_question = self.controller.is_question(user_input)

        # Step 2: Determine input type
        is_important = self.controller.is_important(user_input, is_question)
        is_info, final_prompt = self.controller.check_optional(user_input)


        if not is_question:
            completed, self.controller.current_stage = self.controller.check_stage_completion()


            if completed:
                return "End"
        
        if not is_info:
            if is_important:
                # Step 3: Handle important input
                cot_answer = self.controller.generate_cot_response(user_input, is_question, temp_text)

                final_prompt = self.controller.self_consistency(user_input, cot_answer, 3, temp_text)
            else:
                # Step 4: Handle non-important input
                final_prompt = self.controller.redirect_user(user_input, temp_text)

        # Step 6: Finalize response
        if is_question or is_important or is_info:
            next_steps = self.controller.next_steps(user_input, final_prompt, temp_text)
            final_prompt = final_prompt + "\n\nNext Steps: " + next_steps

        # Step 7: Update conversation history
        self.controller.update_conversation_history(final_prompt, user_input)

        return final_prompt
    
    def more_refine(self, user_input):
        temp_text = self.controller.prepare_conversation_history()
        docs = self.controller.get_docs(user_input, 3)
        is_question = self.controller.is_question(user_input)

        # Step 2: Determine input type
        is_important = self.controller.is_important(user_input, is_question)
        is_info, final_prompt = self.controller.check_optional(user_input)


        if not is_question:
            completed, self.controller.current_stage = self.controller.check_stage_completion()


            if completed:
                return "End"
        
        if not is_info:
            if is_important:
                # Step 3: Handle important input
                cot_answer = self.controller.generate_cot_response(user_input, is_question, temp_text)

                middle_prompt = self.controller.self_consistency(user_input, cot_answer, 3, temp_text)
            else:
                # Step 4: Handle non-important input
                middle_prompt = self.controller.redirect_user(user_input, temp_text)

            # Step 5: Refine response
            feedback = self.controller.feedback(user_input, middle_prompt, temp_text, docs)
            refine = self.controller.refine(middle_prompt, user_input, feedback, temp_text)

            feedback = self.controller.feedback(user_input, refine, temp_text, docs)
            refine = self.controller.refine(refine, user_input, feedback, temp_text)

            feedback = self.controller.feedback(user_input, middle_prompt, temp_text, docs)
            refine = self.controller.refine(middle_prompt, user_input, feedback, temp_text)

            feedback = self.controller.feedback(user_input, refine, temp_text, docs)
            final_prompt = self.controller.refine(refine, user_input, feedback, temp_text)

        # Step 6: Finalize response
        if is_question or is_important or is_info:
            next_steps = self.controller.next_steps(user_input, final_prompt, temp_text)
            final_prompt = final_prompt + "\n\nNext Steps: " + next_steps

        # Step 7: Update conversation history
        self.controller.update_conversation_history(final_prompt, user_input)

        return final_prompt
    
    def no_self_consistency(self, user_input):
        temp_text = self.controller.prepare_conversation_history()
        docs = self.controller.get_docs(user_input, 3)
        is_question = self.controller.is_question(user_input)

        # Step 2: Determine input type
        is_important = self.controller.is_important(user_input, is_question)
        is_info, final_prompt = self.controller.check_optional(user_input)


        if not is_question:
            completed, self.controller.current_stage = self.controller.check_stage_completion()


            if completed:
                return "End"
        
        if not is_info:
            if is_important:
                # Step 3: Handle important input
                middle_prompt = self.controller.generate_cot_response(user_input, is_question, temp_text)
            else:
                # Step 4: Handle non-important input
                middle_prompt = self.controller.redirect_user(user_input, temp_text)

            # Step 5: Refine response
            feedback1 = self.controller.feedback(user_input, middle_prompt, temp_text, docs)
            refine1 = self.controller.refine(middle_prompt, user_input, feedback1, temp_text)

            feedback2 = self.controller.feedback(user_input, refine1, temp_text, docs)
            final_prompt = self.controller.refine(refine1, user_input, feedback2, temp_text)

        # Step 6: Finalize response
        if is_question or is_important or is_info:
            next_steps = self.controller.next_steps(user_input, final_prompt, temp_text)
            final_prompt = final_prompt + "\n\nNext Steps: " + next_steps

        # Step 7: Update conversation history
        self.controller.update_conversation_history(final_prompt, user_input)

        return final_prompt
    
    def custom_llm_setup(self, user_input):
        temp_text = self.controller.prepare_conversation_history()
        docs = ""
        is_question = self.controller.is_question(user_input)

        # Step 2: Determine input type
        is_important = self.controller.is_important(user_input, is_question)
        is_info, final_prompt = self.controller.check_optional(user_input)


        if not is_question:
            completed, self.controller.current_stage = self.controller.check_stage_completion()
            if completed:
                return "End"
            
        
        if not is_info:
            if is_important:
                # Step 3: Handle important input
                middle_prompt = self.controller.generate_cot_response(user_input, is_question, temp_text)

            else:
                # Step 4: Handle non-important input
                middle_prompt = self.controller.redirect_user(user_input, temp_text)

            feedback = self.controller.feedback(user_input, middle_prompt, temp_text, docs)
            final_prompt = self.controller.refine(middle_prompt, user_input, feedback, temp_text)

        # Step 6: Finalize response
        if is_question or is_important or is_info:
            next_steps = self.controller.next_steps(user_input, final_prompt, temp_text)
            final_prompt = final_prompt + "\n\nNext Steps: " + next_steps

        # Step 7: Update conversation history
        self.controller.update_conversation_history(final_prompt, user_input)

        return final_prompt

    # def test_speed(self):
    #     message = "I ask the user the amount of meals eaten using input()"

    #     start_time = time.time()
    #     self.default_llm_setup(message)
    #     end_time = time.time()

    #     elapsed_time = end_time - start_time
    #     print(f"Elapsed time default llm: {elapsed_time:.2f} seconds")

    #     start_time = time.time()
    #     self.no_chat_history_llm_setup(message)
    #     end_time = time.time()
    #     elapsed_time = end_time - start_time
    #     print(f"Elapsed time no chat history llm: {elapsed_time:.2f} seconds")

    #     start_time = time.time()
    #     self.no_rag_history_llm_setup(message)
    #     end_time = time.time()
    #     elapsed_time = end_time - start_time
    #     print(f"Elapsed time no rag history llm: {elapsed_time:.2f} seconds")
        
    #     start_time = time.time()
    #     self.no_redirect(message)
    #     end_time = time.time()
    #     elapsed_time = end_time - start_time
    #     print(f"Elapsed time no redirect: {elapsed_time:.2f} seconds")
    #     time.sleep(20)

    #     start_time = time.time()
    #     self.no_next_steps(message)
    #     end_time = time.time()
    #     elapsed_time = end_time - start_time
    #     print(f"Elapsed time no next steps: {elapsed_time:.2f} seconds")
    #     time.sleep(20)

    #     start_time = time.time()
    #     self.no_CoT(message)
    #     end_time = time.time()
    #     elapsed_time = end_time - start_time
    #     print(f"Elapsed time no CoT: {elapsed_time:.2f} seconds")
    #     time.sleep(20)

    #     start_time = time.time()
    #     self.no_self_refine(message)
    #     end_time = time.time()
    #     elapsed_time = end_time - start_time
    #     print(f"Elapsed time no self refine: {elapsed_time:.2f} seconds")
    #     time.sleep(20)

    #     start_time = time.time()
    #     self.more_refine(message)
    #     end_time = time.time()
    #     elapsed_time = end_time - start_time
    #     print(f"Elapsed time more refine: {elapsed_time:.2f} seconds")
    #     time.sleep(20)

    #     start_time = time.time()
    #     self.no_self_consistency(message)
    #     end_time = time.time()
    #     elapsed_time = end_time - start_time
    #     print(f"Elapsed time no self consistency: {elapsed_time:.2f} seconds")
    #     time.sleep(20)

    #     start_time = time.time()
    #     self.custom_llm_setup(message)
    #     end_time = time.time()
    #     elapsed_time = end_time - start_time
    #     print(f"Elapsed time custom llm setup: {elapsed_time:.2f} seconds")
    #     time.sleep(20)



    def test_LLM_calls(self):

        messages = [
            "What is the circunference of the Earth?",
            "I ask the user the amount of meals eaten using input()",
            "I use a integer to store the value of the user input and use another integer to store the value of the total calories",
            "I don't know what loops are can you explain them to me?",
            "I would use a for loop with the number of meals",
            "I ask the user for input of the calories",
            "I add up all of the calories into a total calories variable",
            "I can give a message based of the amount of calories, like if the amount is over 2000 i will print something like \"to many calories\"",
            "What does the code look like?",
            "I would create a for loop inside the input and do a if that doesn't let values bellow 0"
        ]

        # llms = ["llama3-70b-8192",
        #     "gemma2-9b-it",
        #     "gpt-4o-mini",
        #     "deepseek-r1-distill-llama-70b"
        # ]

        # for llm in llms:
        self.controller.llm_name = "llama3-70b-8192"
        llm = "llama3-70b-8192"


            
        # with open(self.output_file, "a", encoding="utf-8") as file:
        #     file.write(f"------------------------{llm}------------------------\n\n")
        #     file.write("------------------------default_llm_setup------------------------\n\n")

        # self.controller.reset(os.path.join('source', 'Scenarios', 'calory_tracker.json'))

        # print("Starting tests with LLM:", llm)
        # print("default_llm_setup")

        # for user_input in messages:
        #     print(user_input)
        #     message = self.default_llm_setup(user_input)
        #     with open(self.output_file, "a", encoding="utf-8") as file:
        #         file.write("----------------------------------------------------------\n\n")
        #         file.write(f"stage_correct_response_check: {self.controller.stage_correct_response_check}\n")
        #         file.write(f"User Input: {user_input}\n")
        #         file.write(f"LLM Response: {message}\n\n")
        #         file.write("----------------------------------------------------------\n\n")

        # self.controller.reset(os.path.join('source', 'Scenarios', 'calory_tracker.json'))
        # print("no_RAG")

        with open(self.output_file, "a", encoding="utf-8") as file:
            file.write("------------------------no_RAG------------------------\n\n")

        for user_input in messages:
            message = self.no_RAG_llm_setup(user_input)
            with open(self.output_file, "a", encoding="utf-8") as file:
                file.write("----------------------------------------------------------\n\n")
                file.write(f"stage_correct_response_check: {self.controller.stage_correct_response_check}\n")
                file.write(f"User Input: {user_input}\n")
                file.write(f"LLM Response: {message}\n\n")
                file.write("----------------------------------------------------------\n\n")

        # self.controller.reset(os.path.join('source', 'Scenarios', 'calory_tracker.json'))
        # print("no_chat_history_llm_setup")

        # with open(self.output_file, "a", encoding="utf-8") as file:
        #     file.write("------------------------no_chat_history_llm_setup------------------------\n\n")

        # for user_input in messages:
        #     message = self.no_chat_history_llm_setup(user_input)
        #     with open(self.output_file, "a", encoding="utf-8") as file:
        #         file.write("----------------------------------------------------------\n\n")
        #         file.write(f"stage_correct_response_check: {self.controller.stage_correct_response_check}\n")
        #         file.write(f"User Input: {user_input}\n")
        #         file.write(f"LLM Response: {message}\n\n")
        #         file.write("----------------------------------------------------------\n\n")

        # self.controller.reset(os.path.join('source', 'Scenarios', 'calory_tracker.json'))
        # print("no_rag_history_llm_setup")

        # with open(self.output_file, "a", encoding="utf-8") as file:
        #         file.write("-----------------------------no_rag_history_llm_setup-----------------------------\n\n")

        # for user_input in messages:
        #     message = self.no_rag_history_llm_setup(user_input)
        #     with open(self.output_file, "a", encoding="utf-8") as file:
        #         file.write("----------------------------------------------------------\n\n")
        #         file.write(f"stage_correct_response_check: {self.controller.stage_correct_response_check}\n")
        #         file.write(f"User Input: {user_input}\n")
        #         file.write(f"LLM Response: {message}\n\n")
        #         file.write("----------------------------------------------------------\n\n")

        # self.controller.reset(os.path.join('source', 'Scenarios', 'calory_tracker.json'))
        # print("no_redirect")

        # with open(self.output_file, "a", encoding="utf-8") as file:
        #     file.write("------------------------no_redirect------------------------\n\n")

        # for user_input in messages:
        #     message = self.no_redirect(user_input)
        #     with open(self.output_file, "a", encoding="utf-8") as file:
        #         file.write("----------------------------------------------------------\n\n")
        #         file.write(f"stage_correct_response_check: {self.controller.stage_correct_response_check}\n")
        #         file.write(f"User Input: {user_input}\n")
        #         file.write(f"LLM Response: {message}\n\n")
        #         file.write("----------------------------------------------------------\n\n")

        # self.controller.reset(os.path.join('source', 'Scenarios', 'calory_tracker.json'))
        # print("no_next_steps")

        # with open(self.output_file, "a", encoding="utf-8") as file:
        #     file.write("------------------------no_next_steps------------------------\n\n")

        # for user_input in messages:
        #     message = self.no_next_steps(user_input)
        #     with open(self.output_file, "a", encoding="utf-8") as file:
        #         file.write("----------------------------------------------------------\n\n")
        #         file.write(f"stage_correct_response_check: {self.controller.stage_correct_response_check}\n")
        #         file.write(f"User Input: {user_input}\n")
        #         file.write(f"LLM Response: {message}\n\n")
        #         file.write("----------------------------------------------------------\n\n")

        # self.controller.reset(os.path.join('source', 'Scenarios', 'calory_tracker.json'))
        # print("no_CoT")

        # with open(self.output_file, "a", encoding="utf-8") as file:
        #     file.write("------------------------no_CoT------------------------\n\n")

        # for user_input in messages:
        #     message = self.no_CoT(user_input)
        #     with open(self.output_file, "a", encoding="utf-8") as file:
        #         file.write("----------------------------------------------------------\n\n")
        #         file.write(f"stage_correct_response_check: {self.controller.stage_correct_response_check}\n")
        #         file.write(f"User Input: {user_input}\n")
        #         file.write(f"LLM Response: {message}\n\n")
        #         file.write("----------------------------------------------------------\n\n")

        # self.controller.reset(os.path.join('source', 'Scenarios', 'calory_tracker.json'))
        # print("no_self_refine")

        # with open(self.output_file, "a", encoding="utf-8") as file:
        #     file.write("------------------------no_self_refine------------------------\n\n")

        # for user_input in messages:
        #     message = self.no_self_refine(user_input)
        #     with open(self.output_file, "a", encoding="utf-8") as file:
        #         file.write("----------------------------------------------------------\n\n")
        #         file.write(f"stage_correct_response_check: {self.controller.stage_correct_response_check}\n")
        #         file.write(f"User Input: {user_input}\n")
        #         file.write(f"LLM Response: {message}\n\n")
        #         file.write("----------------------------------------------------------\n\n")

        # self.controller.reset(os.path.join('source', 'Scenarios', 'calory_tracker.json'))
        # print("more_refine")

        # with open(self.output_file, "a", encoding="utf-8") as file:
        #     file.write("------------------------more_refine------------------------\n\n")

        # for user_input in messages:
        #     message = self.more_refine(user_input)
        #     with open(self.output_file, "a", encoding="utf-8") as file:
        #         file.write("----------------------------------------------------------\n\n")
        #         file.write(f"stage_correct_response_check: {self.controller.stage_correct_response_check}\n")
        #         file.write(f"User Input: {user_input}\n")
        #         file.write(f"LLM Response: {message}\n\n")
        #         file.write("----------------------------------------------------------\n\n")

        # self.controller.reset(os.path.join('source', 'Scenarios', 'calory_tracker.json'))
        # print("no_self_consistency")

        # with open(self.output_file, "a", encoding="utf-8") as file:
        #     file.write("------------------------no_self_consistency------------------------\n\n")

        # for user_input in messages:
        #     message = self.no_self_consistency(user_input)
        #     with open(self.output_file, "a", encoding="utf-8") as file:
        #         file.write("----------------------------------------------------------\n\n")
        #         file.write(f"stage_correct_response_check: {self.controller.stage_correct_response_check}\n")
        #         file.write(f"User Input: {user_input}\n")
        #         file.write(f"LLM Response: {message}\n\n")
        #         file.write("----------------------------------------------------------\n\n")

        # self.controller.reset(os.path.join('source', 'Scenarios', 'calory_tracker.json'))
        # print("custom_llm_setup")

        # with open(self.output_file, "a", encoding="utf-8") as file:
        #     file.write("------------------------custom_llm_setup------------------------\n\n")

        # for user_input in messages:
        #     message = self.custom_llm_setup(user_input)
        #     with open(self.output_file, "a", encoding="utf-8") as file:
        #         file.write("---------------------------------------------------------\n\n")
        #         file.write(f"stage_correct_response_check: {self.controller.stage_correct_response_check}\n")
        #         file.write(f"User Input: {user_input}\n")
        #         file.write(f"LLM Response: {message}\n\n")
        #         file.write("----------------------------------------------------------\n\n")





if __name__ == "__main__":
    unittest.main()