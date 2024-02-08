import aws_cdk as core
import aws_cdk.assertions as assertions

from acs_fl.acs_fl_stack import AcsFlStack

# example tests. To run these tests, uncomment this file along with the example
# resource in acs_fl/acs_fl_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = AcsFlStack(app, "acs-fl")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
