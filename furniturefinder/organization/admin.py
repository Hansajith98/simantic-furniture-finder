from django.contrib import admin
from organization.models import Organization, Prompt, Configuration, TokenUsage, ProductWebPage

admin.site.register(Organization)
admin.site.register(Prompt)
admin.site.register(Configuration)
admin.site.register(TokenUsage)
admin.site.register(ProductWebPage)
