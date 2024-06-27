import torch
from torch.utils.data import DataLoader
from finetuning_dataloader import CustomDataset
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score

# the evaluation functions need refactoring using our data loader
# the bulk of the code is repetition with data loader stuff, but we've cleaned that up

def evaluate(model, json_data, task, batch_size=32, text_pad_length=500, img_pad_length=36, audio_pad_length=63, shuffle=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    dataset = CustomDataset(json_data, text_pad_length, img_pad_length, audio_pad_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
    model.eval()

    total_loss = 0 
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch_text = batch['text'].to(device)
            batch_text_mask = batch['text_mask'].to(device)
            batch_image = batch['image'].float().to(device)
            batch_mask_img = batch['image_mask'].to(device)
            batch_audio = batch['audio'].float().to(device)
            batch_mask_audio = batch['audio_mask'].to(device)
            batch_pred = batch["binary_label"].to(device) # batch_size by 2

            # batch_size by 2 for binary
            # batch size by 4 by 2 for multi task
            out = model(batch_text, batch_text_mask, batch_image, batch_mask_img, batch_audio, batch_mask_audio, task)
              
            loss = F.binary_cross_entropy(out, batch_pred)
            total_loss += loss.item()

            # Collect predictions and true labels
            preds = (out[:, 1] > 0.5).cpu().numpy()  # Using the second column for binary classification
            true_labels = (batch_pred[:, 1] > 0.5).cpu().numpy()  # Using the second column for binary classification
            
            all_preds.extend(preds)
            all_labels.extend(true_labels)

            if batch_idx == 20:
                break
    
    # Calculate accuracy and F1 score
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')  # use 'macro' or 'weighted' for multi-class

    avg_loss = total_loss / len(dataloader)
    
    print(f'Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
    
    return avg_loss, accuracy, f1



# def multi_evaluate(model, dataset, division):
#     model.eval()

#     total_loss = 0
#     total_loss1, total_loss2, total_loss3 = 0, 0, 0

#     batch_x, batch_y1,batch_image,batch_mask_img = [], [],[],[]
#     batch_director = []
#     batch_genre = []
#     y1_true, y2_true, y3_true = [], [], []
#     imdb_ids = []
#     predictions = [[], [], []]
#     id_to_vec = {}
#     batch_audio, batch_mask_audio = [],[]
#     #batch_audio, batch_mask_audio = [],[]
#     batch_mature, batch_gory, batch_sarcasm, batch_slapstick = [], [], [], []
#     mature_true, gory_true, sarcasm_true, slapstick_true = [], [], [], []
#     batch_text = []
#     predictions_mature, predictions_gory, predictions_slapstick, predictions_sarcasm = [], [], [], []
#     predictions_all, label_true = [], []
#     with torch.no_grad():
#         list_names = []
#         for index,i in enumerate(dataset):
#             mid = dataset[i]['IMDBid']
#             if dataset[i]['label'] == 0 and index != len(dataset) - 1:
#                 continue
            
#             imdb_ids.append(mid)
#             batch_x.append(np.array(dataset[i]['indexes']))

#             file_path = "path_to_I3D_features/"+mid+"_rgb.npy"
#             if not os.path.isfile(file_path): 
#                 count_loop += 1
#                 continue
            
#             path = "path_to_I3D_features/"
#             #image_vec = np.load("./deepMoji_out/"+mid+".npy")
#             a1 = np.load(path+mid+"_rgb.npy")
#             a2 = np.load(path+mid+"_flow.npy")
#             a = a1+a2
#             masked_img = mask_vector(36,a)
#             a = pad_segment(a, 36, 0)
#             image_vec = a
#             batch_image.append(image_vec)
#             #masked_img = mask_vector(36,a)
#             batch_mask_img .append(masked_img)


#             path = "path_to_VGGish_features/"
#             try:
#                 audio_arr = np.load(path+mid+"_vggish.npy")
#             except:
#                 audio_arr = np.array([128*[0.0]])

#             masked_audio = mask_vector(63,audio_arr)
#             #print (masked_audio)
#             audio_vec = pad_segment(audio_arr, 63, 0)
#             batch_audio.append(audio_vec)
#             batch_mask_audio.append(masked_audio)

#             batch_mature.append(dataset[i]['mature'])
#             batch_gory.append(dataset[i]['gory'])
#             batch_slapstick.append(dataset[i]['slapstick'])
#             batch_sarcasm.append(dataset[i]['sarcasm'])

#             mature_label_sample = np.argmax(np.array(dataset[i]['mature']))
#             gory_label_sample = np.argmax(np.array(dataset[i]['gory']))
#             sarcasm_label_sample = np.argmax(np.array(dataset[i]['sarcasm']))
#             slapstick_label_sample = np.argmax(np.array(dataset[i]['slapstick']))
            
#             mature_true.append(mature_label_sample)
#             gory_true.append(gory_label_sample)
#             slapstick_true.append(slapstick_label_sample)
#             sarcasm_true.append(sarcasm_label_sample)

#             label_true.append([mature_label_sample,gory_label_sample,slapstick_label_sample,sarcasm_label_sample])

#             if (len(batch_x) == batch_size or index == len(dataset) - 1) and len(batch_x)>0:

#                 mask = masking(batch_x)

#                 #print (mask)
#                 batch_x = pad_features(batch_x)
#                 batch_x = np.array(batch_x)
#                 batch_x = torch.tensor(batch_x).cuda()

#                 batch_image = np.array(batch_image)
#                 batch_image = torch.tensor(batch_image).cuda()

#                 batch_mask_img = np.array(batch_mask_img )
#                 batch_mask_img = torch.tensor(batch_mask_img ).cuda()

#                 batch_audio = np.array(batch_audio)
#                 batch_audio = torch.tensor(batch_audio).cuda()
 
#                 batch_mask_audio = np.array(batch_mask_audio)
#                 batch_mask_audio = torch.tensor(batch_mask_audio).cuda()

#                 out, mid_level_out = model(batch_x, torch.tensor(mask).cuda(),batch_image.float(),batch_mask_img,batch_audio.float(),batch_mask_audio)

#                 #mature_pred = out[0].cpu()
#                 mature_pred = out[0].cpu()
#                 gory_pred = out[1].cpu()
#                 slapstick_pred = out[2].cpu()
#                 sarcasm_pred = out[3].cpu()

#                 pred_mature = torch.argmax(mature_pred, -1).numpy()
#                 pred_gory = torch.argmax(gory_pred, -1).numpy()
#                 pred_slap = torch.argmax(slapstick_pred, -1).numpy()
#                 pred_sarcasm = torch.argmax(sarcasm_pred, -1).numpy()
                
                
#                 predictions_mature.extend(list(pred_mature))
#                 predictions_gory.extend(list(pred_gory))
#                 predictions_slapstick.extend(list(pred_slap))
#                 predictions_sarcasm.extend(list(pred_sarcasm))

#                 loss2 = F.binary_cross_entropy(mature_pred, torch.Tensor(batch_mature))
#                 # _, labels1 = torch.Tensor(batch_y1).max(dim=1)

#                 total_loss1 += loss2.item()

#                 batch_x, batch_y1,batch_image,batch_mask_img  = [], [], [],[]
#                 batch_director = []
#                 batch_genre = []
#                 batch_mask = []
#                 batch_text = []
#                 batch_similar = []
#                 batch_description = []
#                 imdb_ids = []
#                 batch_audio, batch_mask_audio = [],[]
#                 batch_mature, batch_gory, batch_sarcasm, batch_slapstick = [], [], [], []

#     true_values = []
#     preds = []
#     from sklearn.metrics import hamming_loss
#     for i in range(len(mature_true)):
#          true_values.append([mature_true[i], gory_true[i], slapstick_true[i], sarcasm_true[i]])
#          preds.append([predictions_mature[i],predictions_gory[i],predictions_slapstick[i],predictions_sarcasm[i]])
#     print ("acc_score ",accuracy_score(true_values,preds))
#     print ("Hamin_score",hamming_score(np.array(true_values),np.array( preds)))
#     print("Hamming_loss:", hamming_loss(true_values, preds))
#     print (hamming_loss(true_values, preds) + hamming_score(np.array(true_values),np.array( preds)))
#     from sklearn.metrics import classification_report
#     print (classification_report(true_values,preds))
#     F1_score_mature = f1_score(mature_true, predictions_mature)
#     F1_score_gory = f1_score(gory_true, predictions_gory)
#     F1_score_slap = f1_score(slapstick_true, predictions_slapstick)
#     F1_score_sarcasm = f1_score(sarcasm_true, predictions_sarcasm)
    
#     Average_F1_score = (F1_score_mature + F1_score_gory + F1_score_slap + F1_score_sarcasm)/4
#     print ("Average_F1_score:", Average_F1_score)

#     label_true = np.array(label_true)
#     predictions_all = np.array(predictions_all)
    
#     print('macro All:', f1_score(label_true, predictions_all, average='macro'))
    
#     f = open(path_res_out, "a")
    
#     f.write('macro All: %f\n' % f1_score(label_true, predictions_all, average='macro'))
    
#     print ("Confusion Matrix All:")
#     confusion_matrix_all = multilabel_confusion_matrix(label_true, predictions_all)
#     print (multilabel_confusion_matrix(label_true, predictions_all))
    
#     print ("Mature")
#     print (confusion_matrix(mature_true, predictions_mature))
#     print('weighted', f1_score(mature_true, predictions_mature, average='weighted'))
#     print('micro', f1_score(mature_true, predictions_mature, average='micro'))
#     print('macro', f1_score(mature_true, predictions_mature, average='macro'))
#     print('None', f1_score(mature_true, predictions_mature, average=None))
#     print ("============================")
      
#     f.write ("Mature\n")
#     f.write('weighted: %f\n' % f1_score(mature_true, predictions_mature, average='weighted'))
#     f.write('micro: %f\n' % f1_score(mature_true, predictions_mature, average='micro'))
#     f.write('macro: %f\n' % f1_score(mature_true, predictions_mature, average='macro'))
#     f.write ("============================\n")
    
#     print ("Gory")
#     print (confusion_matrix(gory_true, predictions_gory))
#     print('weighted', f1_score(gory_true, predictions_gory, average='weighted'))
#     print('micro', f1_score(gory_true, predictions_gory, average='micro'))
#     print('macro', f1_score(gory_true, predictions_gory, average='macro'))
#     print('None', f1_score(gory_true, predictions_gory, average=None))
#     print ("=============================")

#     f.write ("Gory\n")
#     f.write('weighted: %f\n' % f1_score(gory_true, predictions_gory, average='weighted'))
#     f.write('micro: %f\n' % f1_score(gory_true, predictions_gory, average='micro'))
#     f.write('macro: %f\n' % f1_score(gory_true, predictions_gory, average='macro'))
#     f.write('binary: %f\n' % f1_score(gory_true, predictions_gory, average='binary'))
#     f.write ("============================\n")
    
#     print ("Slapstick")
#     print (confusion_matrix(slapstick_true, predictions_slapstick))
#     print('weighted', f1_score(slapstick_true, predictions_slapstick, average='weighted'))
#     print('micro', f1_score(slapstick_true, predictions_slapstick, average='micro'))
#     print('macro', f1_score(slapstick_true, predictions_slapstick, average='macro'))
#     print('None', f1_score(slapstick_true, predictions_slapstick, average=None))
#     print ("=============================")

#     f.write ("Slapstick\n")
#     f.write('weighted: %f\n' % f1_score(slapstick_true, predictions_slapstick, average='weighted'))
#     f.write('micro: %f\n' % f1_score(slapstick_true, predictions_slapstick, average='micro'))
#     f.write('macro: %f\n' % f1_score(slapstick_true, predictions_slapstick, average='macro'))
#     f.write('binary: %f\n' % f1_score(slapstick_true, predictions_slapstick, average='binary'))
#     f.write ("============================\n")
   
#     print ("Sarcasm")
#     print (confusion_matrix(sarcasm_true, predictions_sarcasm))
#     print('weighted', f1_score(sarcasm_true, predictions_sarcasm, average='weighted'))
#     print('micro', f1_score(sarcasm_true, predictions_sarcasm, average='micro'))
#     print('macro', f1_score(sarcasm_true, predictions_sarcasm, average='macro'))
#     print('None', f1_score(sarcasm_true, predictions_sarcasm, average=None))
#     print ("=============================")

#     f.write ("Sarcasm\n")
#     f.write('weighted: %f\n' % f1_score(sarcasm_true, predictions_sarcasm, average='weighted'))
#     f.write('micro: %f\n' % f1_score(sarcasm_true, predictions_sarcasm, average='micro'))
#     f.write('macro: %f\n' % f1_score(sarcasm_true, predictions_sarcasm, average='macro'))
#     f.write('binary: %f\n' % f1_score(sarcasm_true, predictions_sarcasm, average='binary'))
#     f.write ("============================\n")
  
#     f.write('acc_score: %f\n' % accuracy_score(true_values,preds))
#     f.write('Hamin_score: %f\n'% hamming_score(np.array(true_values),np.array( preds)))
#     f.write('Hamming_loss: %f\n'% hamming_loss(true_values, preds))
#     f.close()

#     return predictions, \
#            total_loss1 / len(dataset), \
#            F1_score_mature, \
#            F1_score_gory, \
#            F1_score_slap, \
#            F1_score_sarcasm, \
#            Average_F1_score, \
#            confusion_matrix_all, label_true, predictions_all