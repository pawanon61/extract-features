import pandas as pd
import seaborn as sns
sns.reset_orig()
import numpy as np
import argparse

# # TO GET RAW PLOTS FOR EACH NETWORKS ONE AT A TIME
# def get_args():
#     # assign description to help doc
#     parser = argparse.ArgumentParser()
#     # add arguments
#     parser.add_argument('-m', '--modelname', type=str, help='which model do you want to use', required=True)
#     parser.add_argument('-s', '--save', type=bool, default=False, help='if you want to save the plot')
#     # Array for all arguments passed to script
#     args = parser.parse_args()
#     model_arg = args.modelname
#     save_arg = args.save
#     # return all variable values
#     return model_arg, save_arg
    
# model_name, save = get_args()

# df = pd.read_csv('/braintree/home/pgaire/data/csv_output/%s_output.csv' %model_name)
# sns.factorplot(data=df, x='layer', y='value', hue='kind', col='var')
# if not save:
#   sns.plt.show()
# elif save:
#   sns.plt.savefig('/braintree/home/pgaire/data/plots/%s_plot.png' %model_name)







# # TO GET GRAPH FOR ALL NETWORKS AT ONE PLOT
# df = pd.read_csv('/braintree/home/pgaire/data/csv_output/nets_output.csv')

# df = df[(df['var'] == 6) & (df.kind == 'average')]

# sel = df[df.model == 'alexnet']
# sns.plt.plot(np.linspace(0, 1, len(sel)), sel.value)

# sel = df[df.model == 'mobilenet']
# sns.plt.plot(np.linspace(0, 1, len(sel)), sel.value)

# sel = df[df.model == 'vgg16']
# sns.plt.plot(np.linspace(0, 1, len(sel)), sel.value)

# sel = df[df.model == 'vgg19']
# sns.plt.plot(np.linspace(0, 1, len(sel)), sel.value)

# sel = df[df.model == 'resnet50']
# sns.plt.plot(np.linspace(0, 1, len(sel)), sel.value)

# sel = df[df.model == 'xception']
# sns.plt.plot(np.linspace(0, 1, len(sel)), sel.value)

# sel = df[df.model == 'inceptionv1']
# sns.plt.plot(np.linspace(0, 1, len(sel)), sel.value)

# sel = df[df.model == 'inceptionv2']
# sns.plt.plot(np.linspace(0, 1, len(sel)), sel.value)

# sel = df[df.model == 'inceptionv3']
# sns.plt.plot(np.linspace(0, 1, len(sel)), sel.value)

# sel = df[df.model == 'inceptionv4']
# sns.plt.plot(np.linspace(0, 1, len(sel)), sel.value)

# sel = df[df.model == 'inceptionresnetv2']
# sns.plt.plot(np.linspace(0, 1, len(sel)), sel.value)

# sns.plt.legend(['alexnet', 'mobilenet', 'vgg16', 'vgg19', 'resnet50', 'xception', 'inceptionv1', 'inceptionv2', 'inceptionv3', 'inceptionv4', 'inceptionresnetv2'])
# sns.plt.xlabel('depth')
# sns.plt.ylabel('explained variance for var 6 images')

# # sns.plt.show()
# sns.plt.savefig('/braintree/home/pgaire/data/plots/nets_plot.png')






# #TO PLOT TOP 1 VALIDATION ACCURACY VS EXPLAINED VARIANCE
# df = pd.read_csv('/braintree/home/pgaire/data/csv_output/accuracy_and_variance.csv')

# xvalue = df.top_1_validation_accuracy
# yvalue = df.explained_variance_var_6
# annotation_values = ['alexnet', 'mobilenet', 'vgg16', 'resnet50', 'xception', 'inceptionv1', 'inceptionv2', 'inceptionv3', 'inceptionv4', 'inceptionresnetv2']

# fig, ax = sns.plt.subplots()
# ax.scatter(xvalue, yvalue)

# for i, txt in enumerate(annotation_values):
#     ax.annotate(txt, (xvalue[i],yvalue[i]))
# sns.plt.xlabel('top 1 validation accuracy')
# sns.plt.ylabel('explained variance for var 6 images')

# # sns.plt.show()
# sns.plt.savefig('/braintree/home/pgaire/data/plots/accuracy_and_variance.png')








# #TO PLOT TOP 1 VALIDATION ACCURACY VS EXPLAINED VARIANCE
# df = pd.read_csv('/braintree/home/pgaire/data/csv_output/accuracy_and_variance.csv')

# df_average = df[(df.kind=='average')]
# df_average = df_average.set_index([[0,1,2,3,4,5,6,7,8,9]])
# df_100ms = df[(df.kind=='100ms')]
# df_100ms = df_100ms.set_index([[0,1,2,3,4,5,6,7,8,9]])
# df_200ms = df[(df.kind=='200ms')]
# df_200ms = df_200ms.set_index([[0,1,2,3,4,5,6,7,8,9]])


# xvalue_100ms = df_100ms.top_1_validation_accuracy
# yvalue_100ms = df_100ms.explained_variance_var_6
# xvalue_200ms = df_200ms.top_1_validation_accuracy
# yvalue_200ms = df_200ms.explained_variance_var_6
# annotation_values = ['alexnet', 'mobilenet', 'vgg16', 'resnet50', 'xception', 'inceptionv1', 'inceptionv2', 'inceptionv3', 'inceptionv4', 'inceptionresnetv2']

# fig, (ax1, ax2) = sns.plt.subplots(1, 2, sharey=True)
# ax1.scatter(xvalue_100ms, yvalue_100ms, color='blue')

# for i, txt in enumerate(annotation_values):
#     ax1.annotate(txt, (xvalue_100ms[i],yvalue_100ms[i]))

# ax2.scatter(xvalue_200ms, yvalue_200ms, color='red')

# for i, txt in enumerate(annotation_values):
#     ax2.annotate(txt, (xvalue_200ms[i],yvalue_200ms[i]))

# ax1.title.set_text('for 100ms')
# ax2.title.set_text('for 200ms')

# sns.plt.xlabel('top 1 validation accuracy')
# sns.plt.ylabel('explained variance for var 6 images')

# # sns.plt.show()
# sns.plt.savefig('/braintree/home/pgaire/data/plots/accuracy_and_variance.png')










# #PLOT COMPARING TRAINED AND UNTRAINED MODELS

# def get_args():
#     # assign description to help doc
#     parser = argparse.ArgumentParser()
#     # add arguments
#     parser.add_argument('-m', '--modelname', type=str, help='which model do you want to use', required=True)
#     # Array for all arguments passed to script
#     args = parser.parse_args()
#     model_arg = args.modelname
#     # return all variable values
#     return model_arg

# model_name = get_args()

# df_trained = pd.read_csv('/braintree/home/pgaire/data/csv_output/nets_output.csv')
# df_trained['trained?'] = 'yes'
# df_untrained = pd.read_csv('/braintree/home/pgaire/data/csv_output/nets_untrained_output.csv')
# df_untrained['trained?'] = 'no'
# df = pd.concat([df_trained, df_untrained], ignore_index=True)

# fig, (ax1, ax2, ax3) = sns.plt.subplots(1,3, sharey=True)
# d = df[(df['var'] == 6) & (df.kind == 'average')]
# sel = d[(d.model == model_name) & (d['trained?'] == 'no')]
# ax1.plot(np.linspace(0, 1, len(sel)), sel.value)
# sel = d[(d.model == model_name) & (d['trained?'] == 'yes')]
# ax1.plot(np.linspace(0, 1, len(sel)), sel.value)
# ax1.set_title('for average over 70-170 ms')

# d = df[(df['var'] == 6) & (df.kind == '100ms')]
# sel = d[(d.model == model_name) & (d['trained?'] == 'no')]
# ax2.plot(np.linspace(0, 1, len(sel)), sel.value)
# sel = d[(d.model == model_name) & (d['trained?'] == 'yes')]
# ax2.plot(np.linspace(0, 1, len(sel)), sel.value)
# ax2.set_title('for 100ms')

# d = df[(df['var'] == 6) & (df.kind == '200ms')]
# sel = d[(d.model == model_name) & (d['trained?'] == 'no')]
# ax3.plot(np.linspace(0, 1, len(sel)), sel.value)
# sel = d[(d.model == model_name) & (d['trained?'] == 'yes')]
# ax3.plot(np.linspace(0, 1, len(sel)), sel.value)
# ax3.set_title('for 200ms')

# sns.plt.legend(['before training', 'after training'])
# fig.text(0.5, 0.04, 'depth', ha='center', va='center')
# fig.text(0.06, 0.5, 'explained variance for var 6 images', ha='center', va='center', rotation='vertical')
# sns.plt.suptitle(model_name)

# # sns.plt.show()
# sns.plt.savefig('/braintree/home/pgaire/data/plots/before_and_after_training_%s.png' %model_name)









# #PLOT COMPARING ALL TRAINED AND UNTRAINED MODELS IN SINGLE GRAPH

# df_trained = pd.read_csv('/braintree/home/pgaire/data/csv_output/nets_output.csv')
# df_trained['trained?'] = 'yes'
# df_untrained = pd.read_csv('/braintree/home/pgaire/data/csv_output/nets_untrained_output.csv')
# df_untrained['trained?'] = 'no'
# df = pd.concat([df_trained, df_untrained], ignore_index=True)

# model1 = 'alexnet'
# model2 = 'mobilenet'
# model3 = 'resnet50'
# model4 = 'vgg16'
# model5 = 'xception'

# fig, (ax1, ax2, ax3) = sns.plt.subplots(1,3, sharey=True)

# d = df[(df['var'] == 6) & (df.kind == 'average')]
# sel = d[(d.model == model1) & (d['trained?'] == 'no')]
# ax1.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='--', color='red')
# sel = d[(d.model == model1) & (d['trained?'] == 'yes')]
# ax1.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='-', color='red')
# sel = d[(d.model == model2) & (d['trained?'] == 'no')]
# ax1.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='--', color='green')
# sel = d[(d.model == model2) & (d['trained?'] == 'yes')]
# ax1.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='-', color='green')
# sel = d[(d.model == model3) & (d['trained?'] == 'no')]
# ax1.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='--', color='blue')
# sel = d[(d.model == model3) & (d['trained?'] == 'yes')]
# ax1.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='-', color='blue')
# sel = d[(d.model == model4) & (d['trained?'] == 'no')]
# ax1.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='--', color='yellow')
# sel = d[(d.model == model4) & (d['trained?'] == 'yes')]
# ax1.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='-', color='yellow')
# sel = d[(d.model == model5) & (d['trained?'] == 'no')]
# ax1.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='--', color='black')
# sel = d[(d.model == model5) & (d['trained?'] == 'yes')]
# ax1.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='-', color='black')
# ax1.set_title('for average over 70-170 ms')

# d = df[(df['var'] == 6) & (df.kind == '100ms')]
# sel = d[(d.model == model1) & (d['trained?'] == 'no')]
# ax2.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='--', color='red')
# sel = d[(d.model == model1) & (d['trained?'] == 'yes')]
# ax2.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='-', color='red')
# sel = d[(d.model == model2) & (d['trained?'] == 'no')]
# ax2.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='--', color='green')
# sel = d[(d.model == model2) & (d['trained?'] == 'yes')]
# ax2.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='-', color='green')
# sel = d[(d.model == model3) & (d['trained?'] == 'no')]
# ax2.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='--', color='blue')
# sel = d[(d.model == model3) & (d['trained?'] == 'yes')]
# ax2.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='-', color='blue')
# sel = d[(d.model == model4) & (d['trained?'] == 'no')]
# ax2.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='--', color='yellow')
# sel = d[(d.model == model4) & (d['trained?'] == 'yes')]
# ax2.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='-', color='yellow')
# sel = d[(d.model == model5) & (d['trained?'] == 'no')]
# ax2.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='--', color='black')
# sel = d[(d.model == model5) & (d['trained?'] == 'yes')]
# ax2.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='-', color='black')
# ax2.set_title('for 100ms')

# d = df[(df['var'] == 6) & (df.kind == '200ms')]
# sel = d[(d.model == model1) & (d['trained?'] == 'no')]
# ax3.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='--', color='red')
# sel = d[(d.model == model1) & (d['trained?'] == 'yes')]
# ax3.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='-', color='red')
# sel = d[(d.model == model2) & (d['trained?'] == 'no')]
# ax3.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='--', color='green')
# sel = d[(d.model == model2) & (d['trained?'] == 'yes')]
# ax3.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='-', color='green')
# sel = d[(d.model == model3) & (d['trained?'] == 'no')]
# ax3.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='--', color='blue')
# sel = d[(d.model == model3) & (d['trained?'] == 'yes')]
# ax3.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='-', color='blue')
# sel = d[(d.model == model4) & (d['trained?'] == 'no')]
# ax3.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='--', color='yellow')
# sel = d[(d.model == model4) & (d['trained?'] == 'yes')]
# ax3.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='-', color='yellow')
# sel = d[(d.model == model5) & (d['trained?'] == 'no')]
# ax3.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='--', color='black')
# sel = d[(d.model == model5) & (d['trained?'] == 'yes')]
# ax3.plot(np.linspace(0, 1, len(sel)), sel.value, linestyle='-', color='black')
# ax3.set_title('for 200ms')

# import matplotlib.patches as mpatches
# red_patch = mpatches.Patch(color='red', label='alexnet')
# green_patch = mpatches.Patch(color='green', label='mobilenet')
# blue_patch = mpatches.Patch(color='blue', label='resnet50')
# yellow_patch = mpatches.Patch(color='yellow', label='vgg16')
# black_patch = mpatches.Patch(color='black', label='xception')

# sns.plt.legend(handles=[red_patch, green_patch, blue_patch, yellow_patch, black_patch])

# # sns.plt.legend(['before training', 'after training'])
# fig.text(0.5, 0.04, 'depth', ha='center', va='center')
# fig.text(0.06, 0.5, 'explained variance for var 6 images', ha='center', va='center', rotation='vertical')
# sns.plt.suptitle('Before and after training')

# # sns.plt.show()
# sns.plt.savefig('/braintree/home/pgaire/data/plots/before_and_after_training.png')











# # COMPARISION BETWEEN SUPERVISED AND UNSUPERVISED TRAINING

# df_trained = pd.read_csv('/braintree/home/pgaire/data/csv_output/nets_output.csv')
# df_trained = df_trained[(df_trained['var'] == 6) & (df_trained.kind == 'average')]
# df_untrained = pd.read_csv('/braintree/home/pgaire/data/csv_output/nets_untrained_output.csv')
# df_untrained = df_untrained[(df_untrained['var'] == 6) & (df_untrained.kind == 'average')]

# untrained_alexnet_sel = df_untrained[df_untrained.model == 'alexnet']
# sns.plt.plot(np.linspace(0, 1, len(untrained_alexnet_sel)), untrained_alexnet_sel.value)

# trained_alexnet_sel = df_trained[df_trained.model == 'alexnet']
# sns.plt.plot(np.linspace(0, 1, len(trained_alexnet_sel)), trained_alexnet_sel.value)

# trained_slplitbrainautoencoder_sel = df_trained[df_trained.model == 'splitbrainautoencoder']
# sns.plt.plot(np.linspace(0, 0.857, len(trained_slplitbrainautoencoder_sel)), trained_slplitbrainautoencoder_sel.value)

# trained_unsupervisedvideo_sel = df_trained[df_trained.model == 'unsupervisedvideo']
# sns.plt.plot(np.linspace(0, 0.857, len(trained_unsupervisedvideo_sel)), trained_unsupervisedvideo_sel.value)

# sns.plt.legend(['alexnet (untrained)', 'alexnet (supervised)', 'split-brain autoencoder (unsupervised)', 'unsupervised video'])
# sns.plt.xlabel('depth')
# sns.plt.ylabel('explained variance for var 6 images')

# # sns.plt.show()
# sns.plt.savefig('/braintree/home/pgaire/data/plots/supervised_vs_unsupervised.png')






# # TRAINED vs UNTRAINED

# df = pd.read_csv('/braintree/home/pgaire/data/csv_output/untrained_vs_trained.csv')
# df = df[(df.kind == '200ms')]

# xvalue = df.untrained_variance
# yvalue = df.trained_variance
# annotation_values = ['alexnet', 'mobilenet', 'vgg16','resnet50', 'xception']

# fig, ax = sns.plt.subplots()
# ax.scatter(xvalue, yvalue)

# # for i, txt in enumerate(annotation_values):
# #     ax.annotate(txt, (xvalue[i],yvalue[i]))
# sns.plt.xlabel('explained variance for var 6 images for untrained models')
# sns.plt.ylabel('explained variance for var 6 images for trained models')

# sns.plt.show()
# # sns.plt.savefig('/braintree/home/pgaire/data/plots/untrained_vs_trained_200ms.png')